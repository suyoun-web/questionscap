import re
import zipfile
import tempfile
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image

st.set_page_config(page_title="Example 자동 캡쳐 → ZIP", layout="wide")

# ----- allow dash / paren variants -----
DASH_CHARS = r"\-–−－"
RIGHT_PARENS = r"$$$）"

# Example token + id token
EX_WORD_RE = re.compile(r"^Example$$", re.IGNORECASE)
EX_ID_RE = re.compile(rf"^(\d{{1,2}})\.(\d{{1,2}})[{DASH_CHARS}](\d{{1,4}})[{RIGHT_PARENS}]$$")
EX_ID_NOPAREN_RE = re.compile(rf"^(\d{{1,2}})\.(\d{{1,2}})[{DASH_CHARS}](\d{{1,4}})$")

# header/footer hints to cut off
HF_HINT_RE = re.compile(
    r"(YOU,\s*GENIUS|Kakaotalk|Instagram|Phone\s*:|010-\d{3,4}-\d{4}|700\+|MOCK\s*TEST)",
    re.IGNORECASE,
)

# raster crop settings
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

def detect_example_anchors(page, left_ratio=0.80):
    """
    Detect anchors of the form:
      Example 1.1-1)
    Returns list of {id:'1.1-1', y:float}
    """
    w_page = page.rect.width
    words = page.get_text("words") or []
    if not words:
        return []

    anchors = []
    for tokens in group_words_into_lines(words):
        line_text = " ".join(t[4] for t in tokens).strip()

        # skip header/footer-like lines
        if HF_HINT_RE.search(line_text):
            continue

        # keep left-ish lines
        x_left = min(t[0] for t in tokens)
        if x_left > w_page * left_ratio:
            continue

        for i in range(len(tokens)):
            if not EX_WORD_RE.match(norm(tokens[i][4])):
                continue

            # Case 1: next token is full id "1.1-1)"
            if i + 1 < len(tokens):
                t1 = norm(tokens[i+1][4])
                m = EX_ID_RE.match(t1)
                if m:
                    qid = f"{int(m.group(1))}.{int(m.group(2))}-{int(m.group(3))}"
                    anchors.append({"id": qid, "y": tokens[i][1]})
                    break

                # Case 2: "1.1-1" + ")"
                m2 = EX_ID_NOPAREN_RE.match(t1)
                if m2 and i + 2 < len(tokens) and norm(tokens[i+2][4]) == ")":
                    qid = f"{int(m2.group(1))}.{int(m2.group(2))}-{int(m2.group(3))}"
                    anchors.append({"id": qid, "y": tokens[i][1]})
                    break

    anchors.sort(key=lambda d: d["y"])
    return anchors

def find_hf_start_y(page, y_from, y_to):
    ys = []
    for b in page.get_text("blocks"):
        if len(b) < 5:
            continue
        y0 = b[1]
        text = b[4]
        if y0 < y_from or y0 > y_to:
            continue
        if text and HF_HINT_RE.search(str(text)):
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

def build_zip(pdf_bytes, zoom, start_page, end_page, pad_top, pad_bottom, remove_hf=True):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc)
    s = clamp(start_page, 1, n_pages) - 1
    e = clamp(end_page, 1, n_pages) - 1
    if e < s:
        s, e = e, s

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()

    count = 0
    with zipfile.ZipFile(tmp.name, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for pno in range(s, e + 1):
            page = doc[pno]
            w, h = page.rect.width, page.rect.height

            anchors = detect_example_anchors(page)
            if not anchors:
                continue

            for i, a in enumerate(anchors):
                qid = a["id"]
                y0 = a["y"]

                y_start = clamp(y0 - pad_top, 0, h)

                # end at next Example (same page) or end of page
                if i + 1 < len(anchors):
                    next_y = anchors[i + 1]["y"]
                    y_cap = clamp(next_y - 1, 0, h)
                    y_end = clamp(next_y - pad_bottom, y_start + 80, y_cap)
                else:
                    y_cap = h
                    y_end = clamp(h - 8, y_start + 80, h)

                # cut out header/footer blocks if they appear below
                if remove_hf:
                    hf_y = find_hf_start_y(page, y_start, y_cap)
                    if hf_y is not None and hf_y > y_start + 120:
                        y_cap = min(y_cap, hf_y - 4)
                        y_end = min(y_end, y_cap)

                # tighten by ink bbox (includes figures, reduces large blanks)
                scan_clip = fitz.Rect(0, y_start, w, y_end)
                px_bbox = ink_bbox_by_raster(page, scan_clip)
                if px_bbox is not None:
                    tight = px_bbox_to_page_rect(scan_clip, px_bbox, pad_px=INK_PAD_PX)
                    x0 = clamp(tight.x0, 0, w)
                    x1 = clamp(tight.x1, x0 + 80, w)
                    y_end = clamp(tight.y1, y_start + 80, y_end)
                else:
                    x0, x1 = 0, w

                png = render_png(page, fitz.Rect(x0, y_start, x1, y_end), zoom)
                z.writestr(f"{qid}.png", png)
                count += 1

    return tmp.name, count

# ---------------- UI ----------------
st.title("Example 1.1-1) 기준 자동 캡쳐 → ZIP (고정)")

pdf = st.file_uploader("PDF 업로드", type=["pdf"])

colA, colB, colC, colD = st.columns(4)
zoom = colA.slider("해상도(zoom)", 2.0, 4.0, 3.0, 0.1)
start_page = colB.number_input("시작 페이지", min_value=1, value=4, step=1)
end_page = colC.number_input("끝 페이지", min_value=1, value=22, step=1)
remove_hf = colD.checkbox("머릿말/꼬릿말 제거", value=True)

col1, col2 = st.columns(2)
pad_top = col1.slider("위 여백(Example 라인 포함)", 0, 220, 14, 1)
pad_bottom = col2.slider("아래 여백(다음 Example 전)", 0, 220, 12, 1)

if pdf is not None and st.button("생성 & ZIP 다운로드 준비"):
    pdf_bytes = pdf.read()
    base = pdf.name[:-4] if pdf.name.lower().endswith(".pdf") else pdf.name

    with st.spinner("처리 중..."):
        zip_path, count = build_zip(
            pdf_bytes, zoom,
            int(start_page), int(end_page),
            pad_top, pad_bottom,
            remove_hf=remove_hf
        )

    if count == 0:
        st.error("0개 추출됨: 이 범위에 Example 라인이 텍스트로 인식되지 않았을 수 있어요.")
        st.info("Tip: start_page를 3 또는 4로 바꿔보세요.")
    else:
        st.success(f"{count}개 Example 추출 완료")
        with open(zip_path, "rb") as f:
            st.download_button(
                "ZIP 다운로드",
                data=f,
                file_name=f"{base}_Example_p{int(start_page)}-{int(end_page)}.zip",
                mime="application/zip",
            )
