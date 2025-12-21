import re
import zipfile
import tempfile
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image

st.set_page_config(page_title="문제집 자동 캡쳐 → ZIP (패턴 토글)", layout="wide")

# ---------- character variants ----------
DASH_CHARS = r"\-–−－"
RIGHT_PARENS = r"$$$$）"

# ---------- footer/header removal ----------
HF_HINT_RE = re.compile(
    r"(YOU,\s*GENIUS|Kakaotalk|Instagram|Phone\s*:|010-\d{3,4}-\d{4}|700\+|MOCK\s*TEST)",
    re.IGNORECASE,
)

# ---------- raster bbox ----------
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

def find_header_footer_start_y(page, y_from, y_to):
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

# =========================================================
# Anchor detectors (token-based; robust to dash/parens variants)
# Each returns anchors: list of dict {id:str, y:float}
# =========================================================

# 1) Example 1.2-30)
EX_WORD_RE = re.compile(r"^Example$$", re.IGNORECASE)
EX_ID_RE = re.compile(rf"^(\d{{1,2}})\.(\d{{1,2}})[{DASH_CHARS}](\d{{1,4}})[{RIGHT_PARENS}]$$")
EX_ID_NOPAREN_RE = re.compile(rf"^(\d{{1,2}})\.(\d{{1,2}})[{DASH_CHARS}](\d{{1,4}})$$")

def anchors_example(page, left_ratio=0.75):
    w_page = page.rect.width
    words = page.get_text("words") or []
    if not words:
        return []
    anchors = []
    for tokens in group_words_into_lines(words):
        line_text = " ".join(t[4] for t in tokens).strip()
        if HF_HINT_RE.search(line_text):
            continue
        x_left = min(t[0] for t in tokens)
        if x_left > w_page * left_ratio:
            continue

        for i in range(len(tokens)):
            if not EX_WORD_RE.match(norm(tokens[i][4])):
                continue

            if i + 1 < len(tokens):
                t1 = norm(tokens[i+1][4])
                m = EX_ID_RE.match(t1)
                if m:
                    sec, sub, idx = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    anchors.append({"id": f"{sec}.{sub}-{idx}", "y": tokens[i][1]})
                    break

                m2 = EX_ID_NOPAREN_RE.match(t1)
                if m2 and i + 2 < len(tokens) and norm(tokens[i+2][4]) == ")":
                    sec, sub, idx = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
                    anchors.append({"id": f"{sec}.{sub}-{idx}", "y": tokens[i][1]})
                    break
        # no fallback here; keep strict

    anchors.sort(key=lambda d: d["y"])
    return anchors

# 2) 2-8)
AB_RE = re.compile(rf"^(\d{{1,3}})[{DASH_CHARS}](\d{{1,4}})[{RIGHT_PARENS}]$$")
AB_NOPAREN_RE = re.compile(rf"^(\d{{1,3}})[{DASH_CHARS}](\d{{1,4}})$$")

def anchors_ab(page, left_ratio=0.55, max_line_chars=16):
    w_page = page.rect.width
    words = page.get_text("words") or []
    if not words:
        return []
    anchors = []
    for tokens in group_words_into_lines(words):
        line_text = " ".join(t[4] for t in tokens).strip()
        compact = re.sub(r"\s+", "", line_text)
        if HF_HINT_RE.search(line_text):
            continue
        if len(compact) > max_line_chars:
            continue
        x_left = min(t[0] for t in tokens)
        if x_left > w_page * left_ratio:
            continue

        # token match
        found = False
        for (x0,y0,x1,y1,txt) in tokens:
            m = AB_RE.match(norm(txt))
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                anchors.append({"id": f"{a}-{b}", "y": y0})
                found = True
                break
        if found:
            continue

        # split: "2-8" + ")"
        for i in range(len(tokens)-1):
            t1 = norm(tokens[i][4]); t2 = norm(tokens[i+1][4])
            m = AB_NOPAREN_RE.match(t1)
            if m and t2 == ")":
                a, b = int(m.group(1)), int(m.group(2))
                anchors.append({"id": f"{a}-{b}", "y": tokens[i][1]})
                break

    anchors.sort(key=lambda d: d["y"])
    return anchors

# 3) 4)
NP_RE = re.compile(rf"^(\d{{1,4}})[{RIGHT_PARENS}]$$")

def anchors_nparen(page, left_ratio=0.45, max_line_chars=6):
    w_page = page.rect.width
    words = page.get_text("words") or []
    if not words:
        return []
    anchors = []
    for tokens in group_words_into_lines(words):
        line_text = " ".join(t[4] for t in tokens).strip()
        compact = re.sub(r"\s+", "", line_text).replace("）", ")")
        if HF_HINT_RE.search(line_text):
            continue
        if len(compact) > max_line_chars:
            continue
        x_left = min(t[0] for t in tokens)
        if x_left > w_page * left_ratio:
            continue

        m = re.match(r"^(\d{1,4})$$$$$", compact)
        if m:
            anchors.append({"id": m.group(1), "y": min(t[1] for t in tokens)})
    anchors.sort(key=lambda d: d["y"])
    return anchors

# 4) 4.
NDOT_LINE_RE = re.compile(r"^(\d{1,4})\.$$")

def anchors_ndot(page, left_ratio=0.45, max_line_chars=6):
    w_page = page.rect.width
    words = page.get_text("words") or []
    if not words:
        return []
    anchors = []
    for tokens in group_words_into_lines(words):
        line_text = " ".join(t[4] for t in tokens).strip()
        compact = re.sub(r"\s+", "", line_text)
        if HF_HINT_RE.search(line_text):
            continue
        if len(compact) > max_line_chars:
            continue
        x_left = min(t[0] for t in tokens)
        if x_left > w_page * left_ratio:
            continue

        m = NDOT_LINE_RE.match(compact)
        if m:
            anchors.append({"id": m.group(1), "y": min(t[1] for t in tokens)})
    anchors.sort(key=lambda d: d["y"])
    return anchors

DETECTORS = {
    "Example 1.2-30)": anchors_example,
    "2-8)": anchors_ab,
    "4)": anchors_nparen,
    "4.": anchors_ndot,
}

def compute_rects(doc, start_idx, end_idx, detector_name, pad_top, pad_bottom, remove_hf=True):
    rects = []  # list of (page_index, id, rect, page_width)
    detector = DETECTORS[detector_name]

    for pno in range(start_idx, end_idx + 1):
        page = doc[pno]
        w, h = page.rect.width, page.rect.height

        anchors = detector(page)
        if not anchors:
            continue

        for i, a in enumerate(anchors):
            qid = a["id"]
            y0 = a["y"]

            y_start = clamp(y0 - pad_top, 0, h)

            if i + 1 < len(anchors):
                next_y = anchors[i + 1]["y"]
                y_cap = clamp(next_y - 1, 0, h)
                y_end = clamp(next_y - pad_bottom, y_start + 80, y_cap)
            else:
                y_cap = h
                y_end = clamp(h - 8, y_start + 80, h)

            if remove_hf:
                hf_y = find_header_footer_start_y(page, y_start, y_cap)
                if hf_y is not None and hf_y > y_start + 120:
                    y_cap = min(y_cap, hf_y - 4)
                    y_end = min(y_end, y_cap)

            # Ink bbox tighten (figure + remove big blanks)
            scan_clip = fitz.Rect(0, y_start, w, y_end)
            px_bbox = ink_bbox_by_raster(page, scan_clip)
            if px_bbox is not None:
                tight = px_bbox_to_page_rect(scan_clip, px_bbox, pad_px=INK_PAD_PX)
                x0 = clamp(tight.x0, 0, w)
                x1 = clamp(tight.x1, x0 + 80, w)
                y_end = clamp(tight.y1, y_start + 80, y_end)
            else:
                x0, x1 = 0, w

            rects.append((pno, qid, fitz.Rect(x0, y_start, x1, y_end), w))

    return rects

def expand_right_only(rect, target_width, page_width):
    if rect.width >= target_width:
        return rect
    new_x0 = rect.x0
    new_x1 = rect.x0 + target_width
    new_x1 = clamp(new_x1, new_x0 + 80, page_width)
    return fitz.Rect(new_x0, rect.y0, new_x1, rect.y1)

def build_zip(pdf_bytes, zoom, start_page, end_page, detector_name,
              pad_top, pad_bottom, remove_hf=True, unify_width=True):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc)
    s = clamp(start_page, 1, n_pages) - 1
    e = clamp(end_page, 1, n_pages) - 1
    if e < s:
        s, e = e, s

    rects = compute_rects(doc, s, e, detector_name, pad_top, pad_bottom, remove_hf=remove_hf)

    max_width = 0.0
    if unify_width:
        for (_pno, _qid, r, _pw) in rects:
            max_width = max(max_width, r.width)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()

    with zipfile.ZipFile(tmp.name, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for (pno, qid, r, pw) in rects:
            page = doc[pno]
            rect = expand_right_only(r, max_width, pw) if (unify_width and max_width > 0) else r
            png = render_png(page, rect, zoom)
            z.writestr(f"{qid}.png", png)

    return tmp.name, len(rects)

# ---------------- UI ----------------
st.title("문제집 자동 캡쳐 → ZIP (패턴 토글)")

pdf = st.file_uploader("PDF 업로드", type=["pdf"])

colA, colB, colC, colD = st.columns(4)
zoom = colA.slider("해상도(zoom)", 2.0, 4.0, 3.0, 0.1)
start_page = colB.number_input("시작 페이지", min_value=1, value=3, step=1)  # 기본 3
end_page = colC.number_input("끝 페이지", min_value=1, value=22, step=1)
detector_name = colD.selectbox("문제 번호 형식", list(DETECTORS.keys()), index=0)

col1, col2, col3 = st.columns(3)
pad_top = col1.slider("위 여백(번호 라인 포함)", 0, 220, 14, 1)
pad_bottom = col2.slider("아래 여백(다음 번호 전)", 0, 220, 12, 1)
remove_hf = col3.checkbox("머릿말/꼬릿말 제거", value=True)

unify_width = st.checkbox("가로폭 통일(가장 넓은 문제 기준, 오른쪽만 확장)", value=True)

if pdf is not None and st.button("생성 & ZIP 다운로드 준비"):
    pdf_bytes = pdf.read()
    base = pdf.name[:-4] if pdf.name.lower().endswith(".pdf") else pdf.name

    with st.spinner("처리 중..."):
        zip_path, count = build_zip(
            pdf_bytes, zoom,
            int(start_page), int(end_page),
            detector_name,
            pad_top, pad_bottom,
            remove_hf=remove_hf,
            unify_width=unify_width
        )

    if count == 0:
        st.error("0개 추출됨: 선택한 번호 형식이 이 페이지 범위에 없거나, 텍스트 인식이 다를 수 있어요.")
        st.info("팁: 보통 3~4p부터 시작한다면 start_page를 3~4로 두고, 번호 형식을 바꿔가며 테스트하세요.")
    else:
        st.success(f"{count}개 추출 완료")
        with open(zip_path, "rb") as f:
            st.download_button(
                "ZIP 다운로드",
                data=f,
                file_name=f"{base}_{detector_name}_p{int(start_page)}-{int(end_page)}.zip",
                mime="application/zip",
            )
