import re
import io
import zipfile
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image

st.set_page_config(page_title="문제집 PDF → 분류표 폴더별 문제 이미지", layout="wide")

# -------------------------
# 1) 사용자 제공 "대분류(분류표)" : 1~8 -> 폴더명
#    (요청: DAY 폴더 없이 예시2)
# -------------------------
CATEGORY = {
    1: "I_Linear",
    2: "II_Percent",
    3: "III_Unit_conversion",
    4: "IV_Quadratic",
    5: "V_Exponential",
    6: "VI_Polynomials_and_Other_functions",
    7: "VII_Statistics",
    8: "VIII_Geometry",
}

# -------------------------
# 2) 문제 번호 패턴: "1-1)" "8-12)" 등
# -------------------------
QID_RE = re.compile(r"^(\d{1,2})-(\d{1,3})$$$$$")   # 단어 토큰이 정확히 이 형태일 때
DASHNUM_RE = re.compile(r"^(\d{1,2})-(\d{1,3})$$") # "1-1" 과 ")"가 분리되는 PDF 대비

# -------------------------
# 3) 꼬리말(연락처/홍보) 제거 힌트
# -------------------------
FOOTER_HINT_RE = re.compile(
    r"(YOU,\s*GENIUS|700\+\s*MOCK\s*TEST|Kakaotalk|Instagram|010-\d{3,4}-\d{4})",
    re.IGNORECASE,
)

# 잉크(픽셀) 기반 타이트 크롭 설정
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

def detect_question_anchors(page, left_ratio=0.35, max_line_chars=12):
    """
    문제 시작 앵커: '1-1)' 같은 토큰이 거의 단독으로 찍힌 라인을 찾는다.
    반환: list of dict {cat:int, num:int, y:float}
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

        # 꼬리말 라인은 앵커에서 제외
        if FOOTER_HINT_RE.search(line_text):
            continue

        # 너무 긴 라인은 앵커 라인일 가능성 낮음 (하지만 '1-12)'는 길지 않음)
        if len(compact) > max_line_chars:
            continue

        # 왼쪽에 있는 라인만(문제번호는 보통 좌측)
        x_left = min(t[0] for t in tokens)
        if x_left > w_page * left_ratio:
            continue

        cat = num = None
        y_top = None

        # 케이스1: "1-1)" 토큰 그대로
        for (x0, y0, x1, y1, txt) in tokens:
            m = QID_RE.match(txt)
            if m:
                cat = int(m.group(1))
                num = int(m.group(2))
                y_top = y0
                break

        # 케이스2: "1-1" + ")" 분리
        if cat is None:
            for i in range(len(tokens) - 1):
                t1 = tokens[i][4]
                t2 = tokens[i + 1][4]
                m = DASHNUM_RE.match(t1)
                if m and t2 == ")":
                    cat = int(m.group(1))
                    num = int(m.group(2))
                    y_top = tokens[i][1]
                    break

        if cat is None:
            continue

        if cat not in CATEGORY:
            # 1~8 이외면 분류표 밖 → 일단 제외(원하면 OTHER로 넣도록 바꿀 수 있음)
            continue

        anchors.append({"cat": cat, "num": num, "y": y_top})

    anchors.sort(key=lambda d: d["y"])
    return anchors

def find_footer_start_y(page, y_from, y_to):
    """
    y구간 내에서 꼬리말 텍스트 블록이 시작되는 y를 찾는다.
    있으면 그 위에서 잘라내기.
    """
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

def build_zip_from_pdf(pdf_bytes, zoom=3.0, pad_top=12, pad_bottom=12):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    # zip 내부 경로 -> png bytes
    items = []

    for pno in range(len(doc)):
        page = doc[pno]
        w, h = page.rect.width, page.rect.height

        anchors = detect_question_anchors(page)
        if not anchors:
            continue

        for i, a in enumerate(anchors):
            cat = a["cat"]
            num = a["num"]
            y0 = a["y"]

            y_start = clamp(y0 - pad_top, 0, h)

            if i + 1 < len(anchors):
                next_y = anchors[i + 1]["y"]
                y_cap = clamp(next_y - 1, 0, h)
                y_end = clamp(next_y - pad_bottom, y_start + 80, y_cap)
            else:
                y_cap = h
                y_end = clamp(h - 8, y_start + 80, h)

            # 꼬리말 제거 캡
            footer_y = find_footer_start_y(page, y_start, y_cap)
            if footer_y is not None and footer_y > y_start + 120:
                y_cap = min(y_cap, footer_y - 4)
                y_end = min(y_end, y_cap)

            # 잉크 bbox로 좌우/하단 타이트 크롭(그림 포함)
            scan_clip = fitz.Rect(0, y_start, w, y_end)
            px_bbox = ink_bbox_by_raster(page, scan_clip)
            if px_bbox is not None:
                tight = px_bbox_to_page_rect(scan_clip, px_bbox, pad_px=INK_PAD_PX)
                # 번호 포함 위해 y_start는 유지, 아래는 타이트하게
                x0 = clamp(tight.x0, 0, w)
                x1 = clamp(tight.x1, x0 + 80, w)
                new_y_end = clamp(tight.y1, y_start + 80, y_end)
                y_end = new_y_end
            else:
                x0, x1 = 0, w  # fallback

            clip = fitz.Rect(x0, y_start, x1, y_end)
            png = render_png(page, clip, zoom)

            folder = CATEGORY.get(cat, f"{cat}")
            filename = f"{cat}-{num}.png"  # 요청 형식 유지
            arcname = f"{folder}/{filename}"
            items.append((arcname, png))

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for arcname, data in items:
            z.writestr(arcname, data)
    zbuf.seek(0)
    return zbuf, len(items)

st.title("문제집 PDF → 대분류(1~8) 폴더별 문제 이미지 ZIP")

st.write("분류표(고정): 1~8 → I~VIII (DAY 폴더 없이)")

pdf = st.file_uploader("PDF 업로드", type=["pdf"])

col1, col2, col3 = st.columns(3)
zoom = col1.slider("해상도(zoom)", 2.0, 4.0, 3.0, 0.1)
pad_top = col2.slider("위 여백(문제번호 포함)", 0, 80, 12, 1)
pad_bottom = col3.slider("아래 여백(다음 문제 전)", 0, 80, 12, 1)

if pdf is not None:
    pdf_bytes = pdf.read()
    zip_base = pdf.name[:-4] if pdf.name.lower().endswith(".pdf") else pdf.name
    if st.button("생성 & ZIP 다운로드 준비"):
        with st.spinner("자르는 중..."):
            zbuf, count = build_zip_from_pdf(pdf_bytes, zoom=zoom, pad_top=pad_top, pad_bottom=pad_bottom)
        st.success(f"완료: {count}개 문제 이미지 생성")
        st.download_button(
            "ZIP 다운로드",
            data=zbuf,
            file_name=f"{zip_base}.zip",
            mime="application/zip",
        )
else:
    st.info("PDF를 업로드하면 시작합니다.")
