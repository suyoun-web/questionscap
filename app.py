import re
import io
import zipfile
import tempfile
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image

st.set_page_config(page_title="범용 문제집 PDF 크롭 (패턴 선택)", layout="wide")

# -------------------------------------------------------
# 정규식 패턴 모음
# -------------------------------------------------------
PATTERNS = {
    "Type A: 1-1), 8-12) (대분류-번호)": {
        "token": re.compile(r"^(\d{1,2})-(\d{1,3})$$$$"),  # 1-1)
        "split": re.compile(r"^(\d{1,2})-(\d{1,3})$$"),     # 1-1 + )
        "has_category": True
    },
    "Type B: 1., 2. (숫자 + 점)": {
        "token": re.compile(r"^(\d{1,3})\.$$"),             # 1.
        "split": re.compile(r"^(\d{1,3})$$"),               # 1 + .
        "has_category": False
    },
    "Type C: 1), 2) (숫자 + 괄호)": {
        "token": re.compile(r"^(\d{1,3})$$$$"),             # 1)
        "split": re.compile(r"^(\d{1,3})$$"),               # 1 + )
        "has_category": False
    }
}

FOOTER_HINT_RE = re.compile(
    r"(Kakaotalk|Instagram|010-\d{3,4}-\d{4}|YOU,\s*GENIUS|MOCK\s*TEST)",
    re.IGNORECASE,
)

SCAN_ZOOM = 0.6
WHITE_THRESH = 250
INK_PAD_PX = 10

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def group_words_into_lines(words):
    lines = {}
    for w in words:
        x0, y0, x1, y1, txt, bno, lno, wno = w
        lines.setdefault((bno, lno), []).append((x0, y0, x1, y1, txt))
    for k in lines:
        lines[k].sort(key=lambda t: t[0])
    return list(lines.values())

def detect_anchors(page, pattern_key, default_cat=1, left_ratio=0.35, max_line_chars=14):
    """
    선택된 패턴(pattern_key)에 맞는 문제 번호 앵커를 찾는다.
    Type A(1-1)는 텍스트에서 cat 추출, Type B(1.)는 default_cat 사용.
    """
    w_page = page.rect.width
    words = page.get_text("words")
    if not words:
        return []

    lines = group_words_into_lines(words)
    anchors = []
    
    pat_conf = PATTERNS[pattern_key]
    re_token = pat_conf["token"]
    re_split = pat_conf["split"]
    has_cat = pat_conf["has_category"]

    # Type B/C의 분리형(숫자+점) 처리를 위한 보조 문자
    suffix_char = "." if "Type B" in pattern_key else ")"

    for tokens in lines:
        line_text = " ".join(t[4] for t in tokens).strip()
        compact = re.sub(r"\s+", "", line_text)

        if FOOTER_HINT_RE.search(line_text):
            continue
        if len(compact) > max_line_chars:
            continue

        x_left = min(t[0] for t in tokens)
        if x_left > w_page * left_ratio:
            continue

        major = None
        minor = None
        y_top = None

        # Case 1: 토큰 하나에 패턴 매칭 (예: "1-1)" 또는 "1.")
        for (x0, y0, x1, y1, txt) in tokens:
            m = re_token.match(txt)
            if m:
                if has_cat:
                    major = int(m.group(1))
                    minor = int(m.group(2))
                else:
                    major = default_cat
                    minor = int(m.group(1))
                y_top = y0
                break

        # Case 2: 토큰 분리 (예: "1-1" + ")" 또는 "1" + ".")
        if major is None:
            for i in range(len(tokens) - 1):
                t1 = tokens[i][4]
                t2 = tokens[i + 1][4]
                m = re_split.match(t1)
                
                # Type A는 뒤에 )가 와야 함
                # Type B는 뒤에 .가 와야 함
                match_suffix = False
                if has_cat: # Type A
                    if t2 == ")": match_suffix = True
                else:       # Type B/C
                    if t2 == suffix_char: match_suffix = True

                if m and match_suffix:
                    if has_cat:
                        major = int(m.group(1))
                        minor = int(m.group(2))
                    else:
                        major = default_cat
                        minor = int(m.group(1))
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
    if maxx < 0: return None
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
    mp = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"): continue
        if "=" not in line: continue
        k, v = line.split("=", 1)
        if not k.strip().isdigit(): continue
        mp[int(k.strip())] = v.strip().strip("/")
    return mp

def build_zip_streaming(pdf_bytes, zoom, start_page, end_page, pad_top, pad_bottom,
                        mapping, pattern_key, default_cat_val, mode="Accurate", remove_footer=True):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc)
    start = clamp(start_page, 1, n_pages) - 1
    end = clamp(end_page, 1, n_pages) - 1
    if end < start: start, end = end, start

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()

    count = 0
    with zipfile.ZipFile(tmp.name, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for pno in range(start, end + 1):
            page = doc[pno]
            w, h = page.rect.width, page.rect.height

            anchors = detect_anchors(page, pattern_key, default_cat=default_cat_val)
            if not anchors:
                continue

            for i, a in enumerate(anchors):
                major, minor, y0 = a["major"], a["minor"], a["y"]
                
                # 매핑된 폴더명 (없으면 숫자 그대로)
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

                x0, x1 = 0, w
                if mode == "Accurate":
                    scan_clip = fitz.Rect(0, y_start, w, y_end)
                    px_bbox = ink_bbox_by_raster(page, scan_clip)
                    if px_bbox is not None:
                        tight = px_bbox_to_page_rect(scan_clip, px_bbox, pad_px=INK_PAD_PX)
                        x0 = clamp(tight.x0, 0, w)
                        x1 = clamp(tight.x1, x0 + 80, w)
                        y_end = clamp(tight.y1, y_start + 80, y_end)

                clip = fitz.Rect(x0, y_start, x1, y_end)
                png = render_png(page, clip, zoom)

                # 파일명 중복 방지 (Type B는 번호만 있어서 중복 가능성 있음)
                # 안전하게: 번호.png (혹은 번호_p페이지.png)
                # 여기선 사용자 요청인 "번호.png" 유지하되, 폴더 안에서 덮어씌워지는 건 사용자 책임
                # (일반 문제집은 번호가 unique하다고 가정)
                name = f"{minor}.png"
                if "Type A" in pattern_key:
                    name = f"{major}-{minor}.png"
                
                arcname = f"{folder}/{name}"
                z.writestr(arcname, png)
                count += 1

    return tmp.name, count

# ---------------- UI ----------------
st.title("문제집 PDF → 자동 크롭 & 분류")

pdf = st.file_uploader("PDF 업로드", type=["pdf"])

st.divider()
c1, c2 = st.columns(2)

with c1:
    st.subheader("1. 문제 번호 스타일")
    pattern_key = st.radio(
        "형식 선택",
        list(PATTERNS.keys()),
        index=0
    )
    
    default_cat_val = 1
    if "Type A" not in pattern_key:
        st.info("Type B(1.)는 '대분류 번호'가 없으므로 아래 설정된 기본 폴더에 저장됩니다.")
        default_cat_val = st.number_input("기본 폴더 번호 (예: 1 -> I_Linear)", min_value=1, value=1)

with c2:
    st.subheader("2. 폴더(분류표) 설정")
    default_map = "\n".join([
        "1=I_Linear", "2=II_Percent", "3=III_Unit_conversion",
        "4=IV_Quadratic", "5=V_Exponential", "6=VI_Polynomials",
        "7=VII_Statistics", "8=VIII_Geometry"
    ])
    mapping_text = st.text_area("번호=폴더명", value=default_map, height=140)
    mapping = parse_category_mapping(mapping_text)

st.divider()
colA, colB, colC, colD = st.columns(4)
zoom = colA.slider("해상도", 2.0, 4.0, 3.0, 0.1)
start_page = colB.number_input("시작 P", 1, value=1)
end_page = colC.number_input("끝 P", 1, value=50)
mode = colD.selectbox("모드", ["Accurate(정확/느림)", "Fast(빠름)"])

remove_footer = st.checkbox("꼬리말(연락처/홍보) 제거", value=True)

if pdf and st.button("실행"):
    with st.spinner("처리 중..."):
        zip_path, count = build_zip_streaming(
            pdf.read(), zoom, start_page, end_page, 12, 12,
            mapping, pattern_key, default_cat_val,
            mode.split("(")[0], remove_footer
        )
    
    if count == 0:
        st.error(f"찾은 문제가 0개입니다. 선택한 패턴({pattern_key})이 문서 내용과 맞는지 확인하세요.")
    else:
        st.success(f"{count}개 추출 완료!")
        with open(zip_path, "rb") as f:
            st.download_button("ZIP 다운로드", f, file_name="problems.zip", mime="application/zip")
