import re
import zipfile
import tempfile
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image

st.set_page_config(page_title="Example 자동 캡쳐 → ZIP (고정)", layout="wide")

DASHES = ["-", "–", "−", "－"]
RPARENS = [")", "）"]

HF_HINT_RE = re.compile(
    r"(YOU,\s*GENIUS|Kakaotalk|Instagram|Phone\s*:|010-\d{3,4}-\d{4}|700\+|MOCK\s*TEST)",
    re.IGNORECASE,
)

SCAN_ZOOM = 0.6
WHITE_THRESH = 250
INK_PAD_PX = 10

# (1) Example 1.1-1)  형태
EX_TEXT_DOTTED_RE = re.compile(r"Example\s+(\d{1,3})\.(\d{1,3})[-–−－](\d{1,4})\s*[)）]", re.IGNORECASE)
# (2) Example 3-1) 형태 (점 없이)
EX_TEXT_SIMPLE_RE = re.compile(r"Example\s+(\d{1,3})[-–−－](\d{1,4})\s*[)）]", re.IGNORECASE)

def clamp(v, lo, hi): return max(lo, min(hi, v))

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

def detect_example_anchors_text(page):
    txt = page.get_text("text") or ""
    anchors = []

    def find_y_for_example_string(sec, sub, idx, dotted: bool):
        for d in DASHES:
            for rp in RPARENS:
                if dotted:
                    s = f"Example {sec}.{sub}{d}{idx}{rp}"
                else:
                    s = f"Example {sec}{d}{idx}{rp}"
                rects = page.search_for(s)
                if rects:
                    rects.sort(key=lambda r: (r.y0, r.x0))
                    return rects[0].y0
        return None

    for m in EX_TEXT_DOTTED_RE.finditer(txt):
        sec = int(m.group(1)); sub = int(m.group(2)); idx = int(m.group(3))
        qid = f"{sec}.{sub}-{idx}"
        y = find_y_for_example_string(sec, sub, idx, dotted=True)
        if y is not None:
            anchors.append({"id": qid, "y": y})

    for m in EX_TEXT_SIMPLE_RE.finditer(txt):
        sec = int(m.group(1)); idx = int(m.group(2))
        qid = f"{sec}-{idx}"
        y = find_y_for_example_string(sec, None, idx, dotted=False)
        if y is not None:
            anchors.append({"id": qid, "y": y})

    anchors.sort(key=lambda d: d["y"])
    return anchors

def expand_right_only(rect, target_width, page_width):
    if rect.width >= target_width:
        return rect
    new_x0 = rect.x0
    new_x1 = rect.x0 + target_width
    new_x1 = clamp(new_x1, new_x0 + 80, page_width)
    return fitz.Rect(new_x0, rect.y0, new_x1, rect.y1)

def is_mcq_in_band(page, y_from, y_to):
    """
    MCQ 판별: 현재 문제 구간 내에 A) (또는 B)/C)/D))가 있으면 객관식으로 본다.
    """
    clip = fitz.Rect(0, y_from, page.rect.width, y_to)
    t = (page.get_text("text", clip=clip) or "")
    return ("A)" in t) or ("B)" in t) or ("C)" in t) or ("D)" in t)

def compute_rects_for_range(doc, s, e, pad_top, pad_bottom, remove_hf, zoom, frq_space_px):
    rects = []  # (page_index, qid, rect, page_width)
    frq_extra_pt = frq_space_px / zoom  # px -> pt 근사

    for pno in range(s, e + 1):
        page = doc[pno]
        w, h = page.rect.width, page.rect.height

        anchors = detect_example_anchors_text(page)
        if not anchors:
            continue

        for i, a in enumerate(anchors):
            qid = a["id"]
            y0 = a["y"]

            y_start = max(0, y0 - pad_top)

            if i + 1 < len(anchors):
                next_y = anchors[i + 1]["y"]
                y_cap = min(h, next_y - 1)
                y_end = min(y_cap, next_y - pad_bottom)
            else:
                y_cap = h
                y_end = h - 8

            y_end = max(y_start + 80, y_end)

            # 머릿말/꼬릿말이 사이에 끼면 cap 낮추기
            if remove_hf:
                hf_y = find_hf_start_y(page, y_start, y_cap)
                if hf_y is not None and hf_y > y_start + 120:
                    y_cap = min(y_cap, hf_y - 4)
                    y_end = min(y_end, y_cap)

            # 잉크 bbox로 타이트(그림 포함 + 공백 제거)
            scan_clip = fitz.Rect(0, y_start, w, y_end)
            px_bbox = ink_bbox_by_raster(page, scan_clip)
            if px_bbox is not None:
                tight = px_bbox_to_page_rect(scan_clip, px_bbox, pad_px=INK_PAD_PX)
                x0 = max(0, tight.x0)
                x1 = min(w, max(x0 + 80, tight.x1))
                y_end = min(y_end, max(y_start + 80, tight.y1))
            else:
                x0, x1 = 0, w

            # FRQ면 아래 여백 추가(단, y_cap 넘지 않음)
            if frq_space_px > 0 and (not is_mcq_in_band(page, y_start, y_end)):
                y_end = min(y_cap, y_end + frq_extra_pt)

            rects.append((pno, qid, fitz.Rect(x0, y_start, x1, y_end), w))

    return rects

def build_zip(pdf_bytes, zoom, start_page, end_page, pad_top, pad_bottom,
              remove_hf=True, unify_width=True, frq_space_px=250):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc)
    s = clamp(start_page, 1, n_pages) - 1
    e = clamp(end_page, 1, n_pages) - 1
    if e < s:
        s, e = e, s

    rects = compute_rects_for_range(doc, s, e, pad_top, pad_bottom, remove_hf, zoom, frq_space_px)

    max_width = 0.0
    if unify_width:
        for (_pno, _qid, r, _pw) in rects:
            max_width = max(max_width, r.width)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()

    with zipfile.ZipFile(tmp.name, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for (pno, qid, r, page_w) in rects:
            page = doc[pno]
            rect = expand_right_only(r, max_width, page_w) if (unify_width and max_width > 0) else r
            png = render_png(page, rect, zoom)
            z.writestr(f"{qid}.png", png)

    return tmp.name, len(rects)

# ---------------- UI ----------------
st.title("Example (1.1-1) / (3-1) 기준 자동 캡쳐 → ZIP (고정)")

pdf = st.file_uploader("PDF 업로드", type=["pdf"])

colA, colB, colC, colD = st.columns(4)
zoom = colA.slider("해상도(zoom)", 2.0, 4.0, 3.0, 0.1)
start_page = colB.number_input("시작 페이지", min_value=1, value=3, step=1)
end_page = colC.number_input("끝 페이지", min_value=1, value=22, step=1)
remove_hf = colD.checkbox("머릿말/꼬릿말 제거", value=True)

col1, col2, col3, col4 = st.columns(4)
pad_top = col1.slider("위 여백(Example 라인 포함)", 0, 220, 14, 1)
pad_bottom = col2.slider("아래 여백(다음 Example 전)", 0, 220, 12, 1)
unify_width = col3.checkbox("가로폭 최대값에 맞춤(오른쪽만 확장)", value=True)
frq_space_px = col4.slider("FRQ 아래 여백(px)", 0, 600, 250, 25)

if pdf is not None and st.button("생성 & ZIP 다운로드 준비"):
    pdf_bytes = pdf.read()
    base = pdf.name[:-4] if pdf.name.lower().endswith(".pdf") else pdf.name

    with st.spinner("처리 중..."):
        zip_path, count = build_zip(
            pdf_bytes, zoom,
            int(start_page), int(end_page),
            pad_top, pad_bottom,
            remove_hf=remove_hf,
            unify_width=unify_width,
            frq_space_px=frq_space_px
        )

    if count == 0:
        st.error("0개 추출됨: 이 범위에 Example 라인이 텍스트로 매칭되지 않았을 수 있어요.")
    else:
        st.success(f"{count}개 Example 추출 완료")
        with open(zip_path, "rb") as f:
            st.download_button(
                "ZIP 다운로드",
                data=f,
                file_name=f"{base}_Example_p{int(start_page)}-{int(end_page)}.zip",
                mime="application/zip",
            )
