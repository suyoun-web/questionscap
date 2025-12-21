import re
import io
import zipfile
import tempfile
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image

st.set_page_config(page_title="Example 번호로 문제 자동 캡쳐 → ZIP", layout="wide")

# Example 1.1-1) / Example 12.3-45) 등 변형 허용
# - 점(.)과 하이픈(-)은 PDF에 따라 다른 문자일 수 있어 여유 있게 처리
DASH_CHARS = r"\-–−－"
RIGHT_PARENS = r"$$）"

EX_TOKEN_RE = re.compile(
    rf"^Example$$",
    re.IGNORECASE
)
# 다음 토큰(또는 같은 토큰)에 들어갈 번호부: 1.1-1) 형태
EX_ID_RE = re.compile(
    rf"^(\d{{1,2}})\.(\d{{1,2}})[{DASH_CHARS}](\d{{1,4}})[{RIGHT_PARENS}]$$"
)
# "1.1-1" + ")" 분리 대비
EX_ID_NOPAREN_RE = re.compile(
    rf"^(\d{{1,2}})\.(\d{{1,2}})[{DASH_CHARS}](\d{{1,4}})$"
)

# 꼬리말/머리말 힌트(있으면 하단에서 잘라냄)
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

def detect_example_anchors(page, left_ratio=0.60):
    """
    Returns list of dict {sec:int, sub:int, idx:int, y:float}
    Detect patterns like:
      Example 1.1-1)
    where Example and the id may be in same line tokens.
    """
    w_page = page.rect.width
    words = page.get_text("words") or []
    if not words:
        return []

    anchors = []
    for tokens in group_words_into_lines(words):
        line_text = " ".join(t[4] for t in tokens).strip()

        # exclude footer-like lines as anchor lines
        if FOOTER_HINT_RE.search(line_text):
            continue

        x_left = min(t[0] for t in tokens)
        if x_left > w_page * left_ratio:
            continue

        # Scan tokens to find "Example" followed by id
        for i in range(len(tokens)):
            t = norm(tokens[i][4])
            if not EX_TOKEN_RE.match(t):
                continue

            # Case 1: next token is full id "1.1-1)"
            if i + 1 < len(tokens):
                nxt = norm(tokens[i+1][4])
                m = EX_ID_RE.match(nxt)
                if m:
                    sec = int(m.group(1))
                    sub = int(m.group(2))
                    idx = int(m.group(3))
                    y_top = tokens[i][1]
                    anchors.append({"sec": sec, "sub": sub, "idx": idx, "y": y_top})
                    break

                # Case 2: id split "1.1-1" + ")"
                m2 = EX_ID_NOPAREN_RE.match(nxt)
                if m2 and i + 2 < len(tokens):
                    nxt2 = norm(tokens[i+2][4])
                    if nxt2 == ")":
                        sec = int(m2.group(1))
                        sub = int(m2.group(2))
                        idx = int(m2.group(3))
                        y_top = tokens[i][1]
                        anchors.append({"sec": sec, "sub": sub, "idx": idx, "y": y_top})
                        break

            # Case 3: sometimes "Example 1.1-1)" might be a single token (rare)
            # handled by scanning tokens for EX_ID_RE too
        else:
            # Fallback: line contains id token without "Example" (optional)
            for (x0,y0,x1,y1,txt) in tokens:
                m = EX_ID_RE.match(norm(txt))
                if m and ("Example" in line_text or "EXAMPLE" in line_text):
                    anchors.append({"sec": int(m.group(1)), "sub": int(m.group(2)), "idx": int(m.group(3)), "y": y0})
                    break

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

def build_zip(pdf_bytes, zoom, start_page, end_page, pad_top, pad_bottom, remove_footer=True):
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
                sec, sub, idx, y0 = a["sec"], a["sub"], a["idx"], a["y"]

                y_start = clamp(y0 - pad_top, 0, h)
                if i + 1 < len(anchors):
                    next_y = anchors[i + 1]["y"]
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

                # ink bbox tighten (handles figures + big blanks)
                scan_clip = fitz.Rect(0, y_start, w, y_end)
                px_bbox = ink_bbox_by_raster(page, scan_clip)
                if px_bbox is not None:
                    tight = px_bbox_to_page_rect(scan_clip, px_bbox, pad_px=INK_PAD_PX)
                    x0 = clamp(tight.x0, 0, w)
                    x1 = clamp(tight.x1, x0 + 80, w)
                    # keep top, tighten bottom
                    y_end = clamp(tight.y1, y_start + 80, y_end)
                else:
                    x0, x1 = 0, w

                clip = fitz.Rect(x0, y_start, x1, y_end)
                png = render_png(page, clip, zoom)

                filename = f"Example_{sec}.{sub}-{idx}.png"
                z.writestr(filename, png)
                count += 1

    return tmp.name, count

# ---------------- UI ----------------
st.title("200p+ 문제집: Example 1.1-1) 기준 자동 캡쳐 → ZIP")

pdf = st.file_uploader("PDF 업로드", type=["pdf"])

colA, colB, colC, colD = st.columns(4)
zoom = colA.slider("해상도(zoom)", 2.0, 4.0, 3.0, 0.1)
start_page = colB.number_input("시작 페이지", min_value=1, value=1, step=1)
end_page = colC.number_input("끝 페이지", min_value=1, value=20, step=1)
remove_footer = colD.checkbox("꼬리말 제거", value=True)

col1, col2 = st.columns(2)
pad_top = col1.slider("위 여백(Example 라인 포함)", 0, 120, 14, 1)
pad_bottom = col2.slider("아래 여백(다음 Example 전)", 0, 120, 12, 1)

st.caption("권장: 200p+는 1~50, 51~100처럼 페이지 범위로 나눠서 ZIP 다운로드하세요(800문제면 더 안정적).")

if pdf is not None and st.button("생성 & ZIP 다운로드 준비"):
    pdf_bytes = pdf.read()
    zip_base = pdf.name[:-4] if pdf.name.lower().endswith(".pdf") else pdf.name

    with st.spinner("처리 중..."):
        zip_path, count = build_zip(
            pdf_bytes, zoom,
            int(start_page), int(end_page),
            pad_top, pad_bottom,
            remove_footer=remove_footer
        )

    if count == 0:
        st.error("0개 추출됨: 이 범위에 'Example 1.1-1)' 텍스트가 PyMuPDF에서 인식되지 않았을 수 있어요.")
        st.info("확인: PDF에서 'Example 1.1-1)' 텍스트를 드래그 선택/복사할 수 있는지 확인해 주세요.")
    else:
        st.success(f"{count}개 Example 추출 완료")
        with open(zip_path, "rb") as f:
            st.download_button(
                "ZIP 다운로드",
                data=f,
                file_name=f"{zip_base}_p{int(start_page)}-{int(end_page)}.zip",
                mime="application/zip",
            )
