import re
import io
import zipfile
import tempfile
import streamlit as st
import fitz
from PIL import Image

st.set_page_config(page_title="PDF 문제 자동 캡쳐 (혼합 번호 지원)", layout="wide")

# ----- dash / paren variants -----
DASH_CHARS = r"\-–−－"
RIGHT_PARENS = r"$$$）"

# a-b)  (1-1) ...)
AB_TOKEN_RE = re.compile(rf"^(\d{{1,3}})[{DASH_CHARS}](\d{{1,3}})[{RIGHT_PARENS}]$$")
AB_SPLIT_RE = re.compile(rf"^(\d{{1,3}})[{DASH_CHARS}](\d{{1,3}})$$")

# n. (4. 11.)
N_DOT_RE = re.compile(r"^(\d{1,4})\.$$")
N_SPLIT_RE = re.compile(r"^(\d{1,4})$")

DASH_ONLY_RE = re.compile(rf"^[{DASH_CHARS}]$")

FOOTER_HINT_RE = re.compile(
    r"(Kakaotalk|Instagram|010-\d{3,4}-\d{4}|YOU,\s*GENIUS|MOCK\s*TEST|700\+)",
    re.IGNORECASE,
)

SCAN_ZOOM = 0.6
WHITE_THRESH = 250
INK_PAD_PX = 10

def clamp(v, lo, hi): return max(lo, min(hi, v))
def norm(t): return t.replace("）", ")")

def group_words_into_lines(words):
    lines = {}
    for w in words:
        x0,y0,x1,y1,txt,bno,lno,wno = w
        lines.setdefault((bno,lno), []).append((x0,y0,x1,y1,txt))
    for k in lines:
        lines[k].sort(key=lambda t: t[0])
    return list(lines.values())

def detect_anchors_mixed(page, left_ratio=0.50, max_line_chars=32):
    """
    Detect both:
      - a-b)  -> kind="ab", major=a, minor=b
      - n.    -> kind="n",  major=None, minor=n
    Return anchors sorted by y.
    """
    w_page = page.rect.width
    words = page.get_text("words") or []
    if not words:
        return []

    anchors = []
    for tokens in group_words_into_lines(words):
        line_text = " ".join(t[4] for t in tokens).strip()
        compact = re.sub(r"\s+", "", line_text)

        if FOOTER_HINT_RE.search(line_text):
            continue
        if len(compact) > max_line_chars:
            continue

        x_left = min(t[0] for t in tokens)
        if x_left > w_page * left_ratio:
            continue

        # ---- Try a-b) ----
        found = False
        for (x0,y0,x1,y1,txt) in tokens:
            txtn = norm(txt)
            m = AB_TOKEN_RE.match(txtn)
            if m:
                anchors.append({"kind":"ab","major":int(m.group(1)),"minor":int(m.group(2)),"y":y0})
                found = True
                break
        if found:
            continue

        # a-b + )
        for i in range(len(tokens)-1):
            t1 = norm(tokens[i][4])
            t2 = norm(tokens[i+1][4])
            m = AB_SPLIT_RE.match(t1)
            if m and t2 == ")":
                anchors.append({"kind":"ab","major":int(m.group(1)),"minor":int(m.group(2)),"y":tokens[i][1]})
                found = True
                break
        if found:
            continue

        # ---- Try n. ----
        for (x0,y0,x1,y1,txt) in tokens:
            m = N_DOT_RE.match(norm(txt))
            if m:
                anchors.append({"kind":"n","major":None,"minor":int(m.group(1)),"y":y0})
                found = True
                break
        if found:
            continue

        # n + .
        for i in range(len(tokens)-1):
            t1 = norm(tokens[i][4])
            t2 = norm(tokens[i+1][4])
            m = N_SPLIT_RE.match(t1)
            if m and t2 == ".":
                anchors.append({"kind":"n","major":None,"minor":int(m.group(1)),"y":tokens[i][1]})
                found = True
                break

    anchors.sort(key=lambda d: d["y"])
    return anchors

def find_footer_start_y(page, y_from, y_to):
    ys = []
    for b in page.get_text("blocks"):
        if len(b) < 5: 
            continue
        y0 = b[1]; text = b[4]
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
            r,g,b = px[x,y]
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
    maxx = min(w-1, maxx + pad_px)
    maxy = min(h-1, maxy + pad_px)

    x0 = clip.x0 + (minx/(w-1))*(clip.x1-clip.x0)
    x1 = clip.x0 + (maxx/(w-1))*(clip.x1-clip.x0)
    y0 = clip.y0 + (miny/(h-1))*(clip.y1-clip.y0)
    y1 = clip.y0 + (maxy/(h-1))*(clip.y1-clip.y0)
    return fitz.Rect(x0,y0,x1,y1)

def render_png(page, clip, zoom):
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    return pix.tobytes("png")

def parse_mapping(text):
    mp = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"): 
            continue
        if "=" not in line:
            continue
        k,v = line.split("=",1)
        k = k.strip()
        v = v.strip().strip("/")
        if not k.isdigit():
            continue
        mp[int(k)] = v
    return mp

def build_zip(pdf_bytes, zoom, start_page, end_page, pad_top, pad_bottom,
              mapping, mode="Accurate", remove_footer=True,
              n_style_parent="DEFAULT", default_folder_name="DEFAULT"):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc)
    s = clamp(start_page, 1, n_pages)-1
    e = clamp(end_page, 1, n_pages)-1
    if e < s: s,e = e,s

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()

    count = 0
    last_major_seen = None

    with zipfile.ZipFile(tmp.name, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for pno in range(s, e+1):
            page = doc[pno]
            w,h = page.rect.width, page.rect.height

            anchors = detect_anchors_mixed(page)
            if not anchors:
                continue

            for i,a in enumerate(anchors):
                kind = a["kind"]
                major = a["major"]
                minor = a["minor"]
                y0 = a["y"]

                # 폴더 결정
                if kind == "ab":
                    last_major_seen = major
                    folder = mapping.get(major, str(major))
                    fname = f"{major}-{minor}.png"
                else:
                    # n. 형식
                    if n_style_parent == "LAST_MAJOR" and last_major_seen is not None:
                        folder = mapping.get(last_major_seen, str(last_major_seen))
                    else:
                        folder = default_folder_name
                    fname = f"{minor}.png"

                y_start = clamp(y0 - pad_top, 0, h)
                if i+1 < len(anchors):
                    next_y = anchors[i+1]["y"]
                    y_cap = clamp(next_y - 1, 0, h)
                    y_end = clamp(next_y - pad_bottom, y_start + 80, y_cap)
                else:
                    y_cap = h
                    y_end = clamp(h - 8, y_start + 80, h)

                if remove_footer:
                    fy = find_footer_start_y(page, y_start, y_cap)
                    if fy is not None and fy > y_start + 120:
                        y_cap = min(y_cap, fy - 4)
                        y_end = min(y_end, y_cap)

                x0,x1 = 0,w
                if mode == "Accurate":
                    scan_clip = fitz.Rect(0, y_start, w, y_end)
                    px_bbox = ink_bbox_by_raster(page, scan_clip)
                    if px_bbox is not None:
                        tight = px_bbox_to_page_rect(scan_clip, px_bbox)
                        x0 = clamp(tight.x0, 0, w)
                        x1 = clamp(tight.x1, x0 + 80, w)
                        y_end = clamp(tight.y1, y_start + 80, y_end)

                clip = fitz.Rect(x0, y_start, x1, y_end)
                png = render_png(page, clip, zoom)

                z.writestr(f"{folder}/{fname}", png)
                count += 1

    return tmp.name, count

# ---------------- UI ----------------
st.title("문제집 PDF → 혼합 번호(1-1) + (4.) 자동 캡쳐 & 분류")

pdf = st.file_uploader("PDF 업로드", type=["pdf"])

mapping_text = st.text_area(
    "분류표(major=folder). 예: 1=I_Linear",
    value="1=I_Linear\n2=II_Percent\n3=III_Unit_conversion\n4=IV_Quadratic\n5=V_Exponential\n6=VI_Polynomials\n7=VII_Statistics\n8=VIII_Geometry",
    height=120
)
mapping = parse_mapping(mapping_text)

colA,colB,colC,colD = st.columns(4)
zoom = colA.slider("해상도", 2.0, 4.0, 3.0, 0.1)
start_page = colB.number_input("시작 페이지", 1, value=1)
end_page = colC.number_input("끝 페이지", 1, value=10)
mode = colD.selectbox("모드", ["Accurate","Fast"], index=0)

col1,col2,col3 = st.columns(3)
pad_top = col1.slider("위 여백", 0, 80, 12, 1)
pad_bottom = col2.slider("아래 여백", 0, 80, 12, 1)
remove_footer = col3.checkbox("꼬리말 제거", value=True)

st.subheader("n. (예: 4.) 형식 문제를 어디 폴더에 넣을까?")
n_style_parent = st.radio("처리 방식", ["DEFAULT", "LAST_MAJOR"], index=0, horizontal=True)
default_folder_name = st.text_input("DEFAULT 폴더명", value="DEFAULT")

if pdf is not None and st.button("생성 & ZIP 다운로드 준비"):
    pdf_bytes = pdf.read()
    zip_base = pdf.name[:-4] if pdf.name.lower().endswith(".pdf") else pdf.name

    with st.spinner("처리 중..."):
        zip_path, count = build_zip(
            pdf_bytes, zoom, int(start_page), int(end_page),
            pad_top, pad_bottom, mapping,
            mode=mode, remove_footer=remove_footer,
            n_style_parent=n_style_parent, default_folder_name=default_folder_name
        )

    if count == 0:
        st.error("0개 추출됨. 이 페이지 범위에 텍스트로 인식되는 문제번호가 없을 수 있어요.")
        st.info("페이지 범위를 바꿔보거나, PDF에서 번호를 드래그 선택이 되는지 확인해 주세요.")
    else:
        st.success(f"{count}개 추출 완료")
        with open(zip_path, "rb") as f:
            st.download_button(
                "ZIP 다운로드",
                data=f,
                file_name=f"{zip_base}_p{int(start_page)}-{int(end_page)}.zip",
                mime="application/zip",
            )
