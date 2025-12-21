import re
import io
import zipfile
import tempfile
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image

st.set_page_config(page_title="Example 자동 캡쳐 → ZIP (가로폭 통일)", layout="wide")

DASH_CHARS = r"\-–−－"
RIGHT_PARENS = r"$$）"

EX_WORD_RE = re.compile(r"^Example$$", re.IGNORECASE)
EX_ID_RE = re.compile(rf"^(\d{{1,2}})\.(\d{{1,2}})[{DASH_CHARS}](\d{{1,4}})[{RIGHT_PARENS}]$$")
EX_ID_NOPAREN_RE = re.compile(rf"^(\d{{1,2}})\.(\d{{1,2}})[{DASH_CHARS}](\d{{1,4}})$")

HF_HINT_RE = re.compile(
    r"(YOU,\s*GENIUS|Kakaotalk|Instagram|Phone\s*:|010-\d{3,4}-\d{4}|700\+|MOCK\s*TEST)",
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

def detect_example_anchors(page, left_ratio=0.70):
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
                    anchors.append({"sec": int(m.group(1)), "sub": int(m.group(2)), "idx": int(m.group(3)), "y": tokens[i][1]})
                    break

                m2 = EX_ID_NOPAREN_RE.match(t1)
                if m2 and i + 2 < len(tokens) and norm(tokens[i+2][4]) == ")":
                    anchors.append({"sec": int(m2.group(1)), "sub": int(m2.group(2)), "idx": int(m2.group(3)), "y": tokens[i][1]})
                    break

        else:
            if "Example" in line_text or "EXAMPLE" in line_text:
                for (x0,y0,x1,y1,txt) in tokens:
                    m = EX_ID_RE.match(norm(txt))
                    if m:
                        anchors.append({"sec": int(m.group(1)), "sub": int(m.group(2)), "idx": int(m.group(3)), "y": y0})
                        break

    anchors.sort(key=lambda d: d["y"])
    return anchors

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

def expand_right_only(rect, target_width, page_width):
    if rect.width >= target_width:
        return rect
    new_x0 = rect.x0
    new_x1 = rect.x0 + target_width
    new_x1 = clamp(new_x1, new_x0 + 80, page_width)
    return fitz.Rect(new_x0, rect.y0, new_x1, rect.y1)

def compute_rects(doc, start_idx, end_idx, pad_top, pad_bottom, remove_footer=True):
    rects = []  # list of (pno, sec, sub, idx, rect, page_width)
    for pno in range(start_idx, end_idx + 1):
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
                hf_y = find_header_footer_start_y(page, y_start, y_cap)
                if hf_y is not None and hf_y > y_start + 120:
                    y_cap = min(y_cap, hf_y - 4)
                    y_end = min(y_end, y_cap)

            # Ink bbox tighten
            scan_clip = fitz.Rect(0, y_start, w, y_end)
            px_bbox = ink_bbox_by_raster(page, scan_clip)
            if px_bbox is not None:
                tight = px_bbox_to_page_rect(scan_clip, px_bbox, pad_px=INK_PAD_PX)
                x0 = clamp(tight.x0, 0, w)
                x1 = clamp(tight.x1, x0 + 80, w)
                y_end = clamp(tight.y1, y_start + 80, y_end)
            else:
                x0, x1 = 0, w

            rects.append((pno, sec, sub, idx, fitz.Rect(x0, y_start, x1, y_end), w))
    return rects

def build_zip(pdf_bytes, zoom, start_page, end_page, pad_top, pad_bottom, remove_footer=True, unify_width=True):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc)
    s = clamp(start_page, 1, n_pages) - 1
    e = clamp(end_page, 1, n_pages) - 1
    if e < s:
        s, e = e, s

    rects = compute_rects(doc, s, e, pad_top, pad_bottom, remove_footer=remove_footer)

    # 최대 가로폭 계산(전체)
    max_width = 0.0
    if unify_width:
        for (_pno, _sec, _sub, _idx, r, _w) in rects:
            max_width = max(max_width, r.width)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()

    count = 0
    with zipfile.ZipFile(tmp.name, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for (pno, sec, sub, idx, r, page_w) in rects:
            page = doc[pno]
            rect = r
            if unify_width and max_width > 0:
                rect = expand_right_only(rect, max_width, page_w)

            png = render_png(page, rect, zoom)
            filename = f"{sec}.{sub}-{idx}.png"
            z.writestr(filename, png)
            count += 1

    return tmp.name, count

# ---------------- UI ----------------
st.title("200p+ 문제
$$
