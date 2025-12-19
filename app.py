import re
import io
import zipfile
import tempfile
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image

st.set_page_config(page_title="문제집 PDF → 분류 폴더별 문제 이미지", layout="wide")

# ---------- Patterns ----------
# tokens: "1-1)" or "12-103)"
QID_TOKEN_RE = re.compile(r"^(\d{1,2})-(\d{1,3})$$$$")
QID_SPLIT_RE = re.compile(r"^(\d{1,2})-(\d{1,3})$$")  # + next token ")"

# footer hints (optional)
FOOTER_HINT_RE = re.compile(
    r"(Kakaotalk|Instagram|010-\d{3,4}-\d{4}|YOU,\s*GENIUS|MOCK\s*TEST)",
    re.IGNORECASE,
)

# ---------- Raster crop tuning ----------
SCAN_ZOOM = 0.6
WHITE_THRESH = 250
INK_PAD_PX = 10

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def group_words_into_lines(words):
    # words: (x0,y0,x1,y1,txt,block,line,word)
    lines = {}
    for w in words:
        x0, y0, x1, y1, txt, bno, lno, wno = w
        lines.setdefault((bno, lno), []).append((x0, y0, x1, y1, txt))
    for k in lines:
        lines[k].sort(key=lambda t: t[0])
    return list(lines.values())

def detect_anchors(page, left_ratio=0.35, max_line_chars=14):
    """
    Detect "major-minor)" anchors on near-single-token lines at left side.
    Returns: list of dict {major:int, minor:int, y:float}
    """
    w_page = page.rect.width
    words = page.get_text("words")
    if not words:
        return []

    lines = group_words_into_lines(words)
    anchors = []

    for tokens in lines:
        line_text = " ".join(t[4] for t in tokens).strip()
        compact = re.sub(r"\s+", "", line_text)

        # exclude footer-like lines from being anchors
        if FOOTER_HINT_RE.search(line_text):
            continue

        if len(compact) > max_line_chars:
            continue

        x_left = min(t[0] for t in tokens)
        if x_left > w_page * left_ratio:
            continue

        major = minor = None
        y_top = None

        # case1: "1-1)" single token
        for (x0, y0, x1, y1, txt) in tokens:
            m = QID_TOKEN_RE.match(txt)
            if m:
                major = int(m.group(1))
                minor = int(m.group(2))
                y_top = y0
                break

        # case2: "1-1" + ")"
        if major is None:
            for i in range(len(tokens) - 1):
                t1 = tokens[i][4]
                t2 = tokens[i + 1][4]
                m = QID_SPLIT_RE.match(t1)
                if m and t2 == ")":
                    major = int(m.group(1))
                    minor = int(m.group(2))
                    y_top = tokens[i][1]
                    break

        if major is None:
            continue

        anchors.append({"major": major, "minor": minor, "y": y_top})

    anchors.sort(key=lambda d: d["y"])
    return anchors

def find_footer_start_y(page, y_from, y_to):
    ys = []
    for b in page.get_text("blocks"):
        if len(b) < 5:
            continue
        y0 = b[1]
        text = b[4]
        if y0 < y_from or y0 > y_to:
            continue
        if text and FOOTER_HINT_RE.search(str(text)):
            ys.append(y0)
    return min(ys) if ys else None

def ink_bbox_by_raster(page, clip, scan_zoom=SCAN_ZOOM, white_thresh=WHITE_THRESH):
    mat = fitz.Matrix(scan_zoom, scan_zoom)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    w, h = img.size
    px = img.load()
    minx, miny = w, h
    maxx, maxy = -1, -1

    step = 2
    for y in range(0, h, step):
        for x in range(0, w, step):
            r, g, b = px[x, y]
            if r < white_thresh or g < white_thresh or b < white_thresh:
                if x < minx: minx = x
                if y < miny: miny = y
                if x > maxx: maxx = x
                if y > maxy: maxy = y

    if maxx < 0:
        return None
    return (minx, miny, maxx, maxy, w, h)

def px_bbox_to_page_rect(clip, px_bbox, pad_px=INK_PAD_PX):
    minx, miny, maxx, maxy, w, h = px_bbox

    minx = max(0, minx - pad_px)
    miny = max(0, miny - pad_px)
    maxx = min(w - 1, maxx + pad_px)
    maxy = min(h - 1, maxy + pad_px)

    x0 = clip.x0 + (minx / (w - 1)) * (clip.x1 - clip.x0)
    x1 = clip.x0 + (maxx / (w - 1)) * (clip.x1 - clip.x0)
    y0 = clip.y0 + (miny / (h - 1)) * (clip.y1 - clip.y0)
    y1 = clip.y0 + (maxy / (h - 1)) * (clip.y1 - clip.y0)
    return fitz.Rect(x0, y0, x1, y1)

def render_png(page, clip, zoom):
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    return pix.tobytes("png")

def parse_category_mapping(text):
    """
    Text area format:
      1=I_Linear
      2=II_Percent
    Returns: dict[int,str]
    """
    mp = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("/")

        if not k.isdigit():
            continue
        mp[int(k)] = v
    return mp

def build_zip_streaming(pdf_bytes, zoom, start_page, end_page, pad_top, pad_bottom,
                        mapping, mode="Accurate", remove_footer=True):
    """
    Streaming ZIP creation using a temp file (stable for large outputs).
    Returns: (zip_path, count)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc)
    start = clamp(start_page, 1, n_pages) - 1
    end = clamp(end_page, 1, n_pages) - 1
    if end < start:
        start, end = end, start

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()

    count = 0
    with zipfile.ZipFile(tmp.name, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for pno in range(start, end + 1):
            page = doc[pno]
            w, h = page.rect.width, page.rect.height

            anchors = detect_anchors(page)
            if not anchors:
                continue

            for i, a in enumerate(anchors):
                major, minor, y0 = a["major"], a["minor"], a["y"]
                folder = mapping.get(major, f"{major}")

                y_start = clamp(y0 - pad_top, 0, h)

                if i + 1 < len(anchors):
                    next_y = anchors[i + 1]["y"]
                    y_cap = clamp(next_y - 1, 0, h)
                    y_end = clamp(next_y - pad_bottom, y_start + 80, y_cap)
                else:
                    y_cap = h
                    y_end = clamp(h - 8, y_start + 80, h)

                if remove_footer:
                    footer_y = find_footer_start_y(page, y_start, y_cap)
                    if footer_y is not None and footer_y > y_start + 120:
                        y_cap = min(y_cap, footer_y - 4)
                        y_end = min(y_end, y_cap)

                # Base clip: full width between y_start..y_end
                x0, x1 = 0, w

                if mode == "Accurate":
                    scan_clip = fitz.Rect(0, y_start, w, y_end)
                    px_bbox = ink_bbox_by_raster(page, scan_clip)
                    if px_bbox is not None:
                        tight = px_bbox_to_page_rect(scan_clip, px_bbox, pad_px=INK_PAD_PX)
                        x0 = clamp(tight.x0, 0, w)
                        x1 = clamp(tight.x1, x0 + 80, w)
                        # keep top (number line), tighten bottom
                        y_end = clamp(tight.y1, y_start + 80, y_end)

                clip = fitz.Rect(x0, y_start, x1, y_end)
                png = render_png(page, clip, zoom)

                name = f"{major}-{minor}.png"
                arcname = f"{folder}/{name}"
                z.writestr(arcname, png)
                count += 1

    return tmp.name, count

# ---------------- UI ----------------
st.title("문제집 PDF → (대분류-번호) 기준 폴더 분류 ZIP")
st.caption("문제번호 형식: 1-1), 1-2) … / 분류표는 앱에서 직접 입력 (문제집마다 변경 가능)")
st.divider()

pdf = st.file_uploader("PDF 업로드", type=["pdf"])

colA, colB, colC, colD = st.columns(4)
zoom = colA.slider("해상도(zoom)", 2.0, 4.0, 3.0, 0.1)
start_page = colB.number_input("시작 페이지", min_value=1, value=1, step=1)
end_page = colC.number_input("끝 페이지", min_value=1, value=50, step=1)
mode = colD.selectbox("모드", ["Accurate", "Fast"], index=0)

col1, col2, col3 = st.columns(3)
pad_top = col1.slider("위 여백(문제번호 포함)", 0, 100, 12, 1)
pad_bottom = col2.slider("아래 여백(다음 문제 전)", 0, 100, 12, 1)
remove_footer = col3.checkbox("꼬리말(연락처/홍보) 제거", value=True)

default_map = "\n".join([
    "# 형식: 대분류번호=폴더명",
    "1=I_Linear",
    "2=II_Percent",
    "3=III_Unit_conversion",
    "4=IV_Quadratic",
    "5=V_Exponential",
    "6=VI_Polynomials_and_Other_functions",
    "7=VII_Statistics",
    "8=VIII_Geometry",
])

mapping_text = st.text_area("분류표 입력(문제집마다 바꿔서 사용)", value=default_map, height=180)
mapping = parse_category_mapping(mapping_text)

if pdf is not None:
    pdf_bytes = pdf.read()
    zip_base = pdf.name[:-4] if pdf.name.lower().endswith(".pdf") else pdf.name

    if st.button("생성 & ZIP 다운로드 준비"):
        with st.spinner("처리 중... (페이지 수/문제 수에 따라 시간이 걸릴 수 있어요)"):
            zip_path, count = build_zip_streaming(
                pdf_bytes=pdf_bytes,
                zoom=zoom,
                start_page=int(start_page),
                end_page=int(end_page),
                pad_top=pad_top,
                pad_bottom=pad_bottom,
                mapping=mapping,
                mode=mode,
                remove_footer=remove_footer,
            )

        with open(zip_path, "rb") as f:
            st.success(f"완료: {count}개 문제 이미지 생성")
            st.download_button(
                "ZIP 다운로드",
                data=f,
                file_name=f"{zip_base}_p{int(start_page)}-{int(end_page)}.zip",
                mime="application/zip",
            )
else:
    st.info("PDF를 업로드하세요. 800문제급이면 1~50, 51~100처럼 페이지 범위로 나눠서 다운로드하는 걸 권장합니다.")
