import re
import io
import zipfile
import tempfile
import streamlit as st
import fitz
from PIL import Image

st.set_page_config(page_title="Example 자동 캡쳐 → ZIP (text 기반)", layout="wide")

DASH_CHARS = "-–−－"
RIGHT_PARENS = ")）"

HF_HINT_RE = re.compile(
    r"(YOU,\s*GENIUS|Kakaotalk|Instagram|Phone\s*:|010-\d{3,4}-\d{4}|700\+|MOCK\s*TEST)",
    re.IGNORECASE,
)

SCAN_ZOOM = 0.6
WHITE_THRESH = 250
INK_PAD_PX = 10

# text에서 Example id를 뽑는 정규식(다양한 대시/괄호 허용)
EX_TEXT_RE = re.compile(
    rf"Example\s+(\d{{1,2}})\.(\d{{1,2}})[{re.escape(DASH_CHARS)}](\d{{1,4}})\s*[{re.escape(RIGHT_PARENS)}]",
    re.IGNORECASE
)

def clamp(v, lo, hi): return max(lo, min(hi, v))
def norm_paren(s): return s.replace("）", ")")

def find_footer_start_y(page, y_from, y_to):
    ys = []
    for b in page.get_text("blocks"):
        if len(b) < 5: 
            continue
        y0 = b[1]; text = b[4]
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
    """
    1) page.get_text("text")에서 Example id들을 regex로 찾고
    2) 각 매치 문자열을 page.search_for로 찾아 y를 얻는다.
    """
    txt = page.get_text("text") or ""
    anchors = []

    for m in EX_TEXT_RE.finditer(txt):
        sec = int(m.group(1))
        sub = int(m.group(2))
        idx = int(m.group(3))

        # search_for는 정확 매칭이 중요해서, "Example {sec}.{sub}-{idx})" 형태로 몇 가지 시도
        candidates = []
        for dash in ["-", "–", "−", "－"]:
            for rp in [")", "）"]:
                candidates.append(f"Example {sec}.{sub}{dash}{idx}{rp}")

        found_rect = None
        for s in candidates:
            rects = page.search_for(s)
            if rects:
                # 가장 왼쪽/위쪽의 것 선택
                rects.sort(key=lambda r: (r.y0, r.x0))
                found_rect = rects[0]
                break

        if found_rect is None:
            # search_for 실패 시, fallback: y는 못 얻으니 스킵
            continue

        anchors.append({"sec": sec, "sub": sub, "idx": idx, "y": found_rect.y0})

    anchors.sort(key=lambda d: d["y"])
    return anchors

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

            anchors = detect_example_anchors_text(page)
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

                scan_clip = fitz.Rect(0, y_start, w, y_end)
                px_bbox = ink_bbox_by_raster(page, scan_clip)
                if px_bbox is not None:
                    tight = px_bbox_to_page_rect(scan_clip, px_bbox, pad_px=INK_PAD_PX)
                    x0 = clamp(tight.x0, 0, w)
                    x1 = clamp(tight.x1, x0 + 80, w)
                    y_end = clamp(tight.y1, y_start + 80, y_end)
                else:
                    x0, x1 = 0, w

                clip = fitz.Rect(x0, y_start, x1, y_end)
                png = render_png(page, clip, zoom)

                filename = f"{sec}.{sub}-{idx}.png"
                z.writestr(filename, png)
                count += 1

    return tmp.name, count

st.title("200p+ 문제집: Example 1.1-1) 기준 자동 캡쳐 → ZIP (text 기반)")

pdf = st.file_uploader("PDF 업로드", type=["pdf"])

colA, colB, colC = st.columns(3)
zoom = colA.slider("해상도(zoom)", 2.0, 4.0, 3.0, 0.1)
start_page = colB.number_input("시작 페이지", min_value=1, value=1, step=1)
end_page = colC.number_input("끝 페이지", min_value=1, value=22, step=1)

col1, col2, col3 = st.columns(3)
pad_top = col1.slider("위 여백(Example 라인 포함)", 0, 200, 14, 1)
pad_bottom = col2.slider("아래 여백(다음 Example 전)", 0, 200, 12, 1)
remove_footer = col3.checkbox("머릿말/꼬릿말 제거", value=True)

if pdf is not None and st.button("생성 & ZIP 다운로드 준비"):
    pdf_bytes = pdf.read()
    zip_base = pdf.name[:-4] if pdf.name.lower().endswith(".pdf") else pdf.name

    with st.spinner("처리 중..."):
        zip_path, count = build_zip(pdf_bytes, zoom, int(start_page), int(end_page), pad_top, pad_bottom, remove_footer=remove_footer)

    if count == 0:
        st.error("0개 추출됨: page.search_for가 문자열을 못 찾는 케이스일 수 있어요(특수문자/공백/분리).")
        st.info("다음 단계: 디버그로 page.get_text('text')의 Example 라인을 그대로 출력해 패턴을 더 정확히 맞출 수 있어요.")
    else:
        st.success(f"{count}개 Example 추출 완료")
        with open(zip_path, "rb") as f:
            st.download_button(
                "ZIP 다운로드",
                data=f,
                file_name=f"{zip_base}_p{int(start_page)}-{int(end_page)}.zip",
                mime="application/zip",
            )
