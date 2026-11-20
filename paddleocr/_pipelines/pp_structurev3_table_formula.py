# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PP-StructureV3 pipeline variant with formula recognition inside tables.

The standard PP-StructureV3 pipeline never sends table cell content to the
formula recognition model: the page-level layout model only labels regions
*outside* tables, and cell content is handled by the table recognition
(OCR + structure) sub-pipeline. This module provides
:class:`PPStructureV3TableCellFormula`, a drop-in subclass that adds a
post-processing stage to ``predict``:

1. crop every table cell and run layout detection on it;
2. sub-regions labeled ``formula`` are recognized line by line with the
   formula recognition model (fragments of one formula on the same visual
   line are merged first, and formula-like OCR text gaps missed by the
   layout model are recovered, trimmed to their formula core so plain text
   at the gap edges stays on the text side);
3. plain-text stretches between the formula regions are re-recognized from
   the image pixels (the line-level OCR reading cannot simply be cut at
   the region boundaries), and the ``<td>`` content of ``pred_html`` is
   rebuilt in reading order with recognized formulas wrapped in
   ``$$...$$``.
"""

import logging
import re
import unicodedata
from pathlib import Path

from paddlex import create_model

from .pp_structurev3 import PPStructureV3
from .._models.formula_recognition import FormulaRecognition
from .._models.text_recognition import TextRecognition

_logger = logging.getLogger(__name__)

DEFAULT_CELL_LAYOUT_MODEL = "PP-DocLayout_plus-L"
DEFAULT_FORMULA_MODEL = "PP-FormulaNet_plus-M"
DEFAULT_CELL_TEXT_REC_MODEL = "PP-OCRv5_server_rec"

# Sentence punctuation that can sit at the boundary between plain text and a
# formula region inside a mixed text line. Cutting a line at a formula-region
# boundary (or re-recognizing the cut stretch) may silently drop such
# punctuation, so the per-cell invariant checks it is preserved.
_BOUNDARY_PUNCT = "：:；;，,、。．.!！？?"


def _poly_center(poly):
    return (
        (poly[:, 0].min() + poly[:, 0].max()) / 2,
        (poly[:, 1].min() + poly[:, 1].max()) / 2,
    )


def _poly_box(poly):
    return [
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    ]


def _char_spans(text, tx1, tx2):
    """Estimate the x interval of each character in a text line.

    Positions are proportional to character widths: a full-width
    (e.g. CJK) character counts twice a half-width one. This keeps the
    estimated boundaries usable on lines mixing Chinese and Latin/digit
    characters, unlike a plain uniform mapping.
    """
    weights = [
        1.0 if unicodedata.east_asian_width(ch) in ("W", "F") else 0.5
        for ch in text
    ]
    total = sum(weights) or float(len(weights))
    scale = (tx2 - tx1) / total
    spans = []
    acc = tx1
    for wgt in weights:
        spans.append((acc, acc + wgt * scale))
        acc += wgt * scale
    return spans


def _intersection_area(box_a, box_b):
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _dedup_regions(regions, overlap_ratio=0.5):
    """NMS-like dedup so the same content is never recognized twice."""
    regions = sorted(
        regions,
        key=lambda r: (r[2] - r[0]) * (r[3] - r[1]),
        reverse=True,
    )
    kept = []
    for region in regions:
        area = (region[2] - region[0]) * (region[3] - region[1])
        duplicate = any(
            _intersection_area(region, kept_region) / area > overlap_ratio
            for kept_region in kept
        )
        if not duplicate:
            kept.append(region)
    # Restore reading order (top-to-bottom, left-to-right).
    return sorted(kept, key=lambda r: (r[1], r[0]))


def _merge_adjacent_regions(regions, max_gap_ratio=1.2):
    """Merge formula fragments of the same formula on one visual line.

    Two regions are merged when they overlap vertically by at least half
    of the smaller height (same visual line) and their horizontal gap is
    at most ``max_gap_ratio`` times that height, e.g. "Q₃=2" + "5m³/h"
    in "Q₃=2.5m³/h". Merging makes the formula model see the complete
    formula instead of disjoint fragments.
    """
    merged = []
    for region in sorted(regions, key=lambda r: (r[1], r[0])):
        rx1, ry1, rx2, ry2 = region
        target = None
        for m in merged:
            mx1, my1, mx2, my2 = m
            oy1, oy2 = max(ry1, my1), min(ry2, my2)
            min_h = min(ry2 - ry1, my2 - my1)
            if min_h <= 0 or oy2 <= oy1:
                continue
            if (oy2 - oy1) / min_h < 0.5:
                continue
            if rx1 >= mx2:
                gap = rx1 - mx2
            elif mx1 >= rx2:
                gap = mx1 - rx2
            else:
                gap = 0.0
            if gap <= max_gap_ratio * min_h:
                target = m
                break
        if target is not None:
            target[0] = min(target[0], rx1)
            target[1] = min(target[1], ry1)
            target[2] = max(target[2], rx2)
            target[3] = max(target[3], ry2)
        else:
            merged.append(list(region))
    # Restore reading order (top-to-bottom, left-to-right).
    return sorted(merged, key=lambda r: (r[1], r[0]))


def _looks_like_formula(text):
    """Heuristic: does this OCR fragment look like formula content?

    Matches fragments containing math operators, superscript/subscript
    digits, or decimal continuations like ".5" — the "." in a numeric
    context is a decimal point, not a sentence separator.
    """
    t = text.strip().strip(";；,，、")
    if not t:
        return False
    if any(ch in "=+×÷/³²△Δ±" for ch in t):
        return True
    if any("\u2080" <= ch <= "\u2089" or "\u2070" <= ch <= "\u2079" for ch in t):
        return True
    return bool(re.match(r"^\.\d", t))


def _substring_for_interval(text, char_spans, ix1, ix2):
    """Fallback: estimate the substring of ``text`` lying in [ix1, ix2]."""
    s = sum(1 for cx1, cx2 in char_spans if (cx1 + cx2) / 2 < ix1)
    e = sum(1 for cx1, cx2 in char_spans if (cx1 + cx2) / 2 < ix2)
    return text[s:e].strip()


def _reconcile_boundary_punct(rec_text, fallback):
    """Trust the whole-line OCR reading for boundary punctuation that the
    re-recognition of a narrow crop dropped: prepend/append punctuation
    marks present at the edges of the estimated fallback substring but
    missing from the recognized text (compared after NFKC folding, so
    "，" matches ","). E.g. the 20-pixel-wide crop of ";D" being
    recognized as just "D".
    """
    rec, fb = rec_text, fallback

    def _fold(ch):
        return unicodedata.normalize("NFKC", ch)

    while fb and fb[0] in _BOUNDARY_PUNCT and (
        not rec or _fold(rec[0]) != _fold(fb[0])
    ):
        rec = fb[0] + rec
        fb = fb[1:]
    while fb and fb[-1] in _BOUNDARY_PUNCT and (
        not rec or _fold(rec[-1]) != _fold(fb[-1])
    ):
        rec = rec + fb[-1]
        fb = fb[:-1]
    return rec


def _split_text_line(
    text, text_box, regions, doc_img, text_recognizer,
    min_overlap_ratio=0.5,
):
    """Split a text line at the formula regions, keeping the plain-text
    stretches between/around them.

    The line-level OCR text cannot simply be cut at the region
    boundaries: character positions are only estimated proportionally
    to character widths (so a character can be mis-mapped across a
    boundary, e.g. the "Q" of "Q₃=..." leaking into the kept text even
    though the formula crop already contains it), and the OCR itself
    may have mis-read the line (e.g. "N20，Q₃=4.0m³/h;DN" recognized as
    "N20Q₃=4.m³/h;DN", silently dropping the "，"). So each plain-text
    stretch is cropped from the image and re-recognized with the text
    recognition model, and the estimated substring is only used as a
    fallback when re-recognition returns nothing.

    A region splits the line only if it overlaps it vertically by at
    least ``min_overlap_ratio`` of the line height, so text next to
    (not under) a tall formula box is preserved.

    Returns a list of ``(y_center, x_center, kind, text, height)`` items.
    """
    tx1, ty1, tx2, ty2 = text_box
    if not text:
        return []
    line_h = max(ty2 - ty1, 1.0)
    covered = []
    for rx1, ry1, rx2, ry2 in regions:
        iy1, iy2 = max(ry1, ty1), min(ry2, ty2)
        if iy2 <= iy1 or (iy2 - iy1) / line_h < min_overlap_ratio:
            continue
        covered.append((max(rx1, tx1), min(rx2, tx2)))

    # Plain-text pixel intervals: the complement of the covered
    # intervals within the line's x range.
    intervals = []
    cursor = tx1
    for cx1, cx2 in sorted(covered):
        if cx1 > cursor:
            intervals.append((cursor, cx1))
        cursor = max(cursor, cx2)
    if cursor < tx2:
        intervals.append((cursor, tx2))

    char_spans = _char_spans(text, tx1, tx2)
    img_h, img_w = doc_img.shape[:2]
    line_cy = (ty1 + ty2) / 2
    # Stretches narrower than a character are slivers cut by an
    # imprecise region boundary (at most a fraction of one glyph);
    # recognizing them only yields garbage, so skip them outright.
    min_seg_width = max(10.0, 0.3 * line_h)
    out = []
    for ix1, ix2 in intervals:
        sub = _substring_for_interval(text, char_spans, ix1, ix2)
        # Re-recognize the plain-text stretch from the image pixels so
        # the kept text reflects what is actually printed between the
        # formula regions, not the error-prone whole-line OCR reading.
        pad_y = 2
        cy1 = max(0, int(round(ty1)) - pad_y)
        cy2 = min(img_h, int(round(ty2)) + pad_y)
        cx1 = max(0, int(round(ix1)))
        cx2 = min(img_w, int(round(ix2)))
        if cx2 - cx1 >= min_seg_width and cy2 - cy1 >= min_seg_width:
            seg_crop = doc_img[cy1:cy2, cx1:cx2]
            rec_result = list(text_recognizer.predict(seg_crop))
            rec_text = rec_result[0]["rec_text"].strip()
            if rec_text:
                # A narrow crop may lose its leading/trailing sentence
                # punctuation; take it back from the whole-line OCR.
                sub = _reconcile_boundary_punct(rec_text, sub)
        if sub:
            out.append((line_cy, (ix1 + ix2) / 2, "text", sub, line_h))
    return out


def _group_into_lines(items):
    """Group items into visual lines by their vertical centers.

    Items are ``(y_center, x_center, kind, content, height)`` tuples.
    Items whose vertical center is within ``0.6 * median height`` of the
    first item of a line belong to the same visual line; each line is
    then sorted left-to-right.
    """
    if not items:
        return []
    heights = sorted(item[4] for item in items)
    tol = max(10.0, 0.6 * heights[len(heights) // 2])
    items = sorted(items, key=lambda item: item[0])
    lines = []
    for item in items:
        if lines and item[0] - lines[-1][0][0] <= tol:
            lines[-1].append(item)
        else:
            lines.append([item])
    for line in lines:
        line.sort(key=lambda item: item[1])
    return lines


def _mathy_char(ch):
    return (
        ch in "=+×÷/³²△Δ±"
        or "\u2080" <= ch <= "\u2089"
        or "\u2070" <= ch <= "\u2079"
    )


def _trim_gap_to_formula_core(text, char_spans, ix1, ix2):
    """Trim plain-text edges off a promoted formula-like text gap.

    A gap is promoted as a whole when any part of it looks like formula
    content, so plain text at its edges is wrongly included (e.g. the
    ";D" of "…Q₃=2.5m³/h;D" where the line wraps "DN" across lines as
    "…;D / N20…") and then leaks into the recognized LaTeX. Cut the
    interval back to its formula core: at each edge, when the run of
    non-math characters contains sentence punctuation, everything from
    that punctuation outward is plain text.
    """
    inside = [
        i
        for i, (a, b) in enumerate(char_spans)
        if ix1 <= (a + b) / 2 < ix2
    ]
    if not inside:
        return ix1, ix2
    # Trailing edge: non-math run from the end; cut at the first
    # (leftmost) boundary punctuation inside the run.
    j = len(inside) - 1
    while j >= 0 and not _mathy_char(text[inside[j]]):
        j -= 1
    for k in inside[j + 1:]:
        if text[k] in _BOUNDARY_PUNCT:
            ix2 = min(ix2, char_spans[k][0])
            break
    # Leading edge: non-math run from the start; cut at the last
    # (rightmost) boundary punctuation inside the run.
    j = 0
    while j < len(inside) and not _mathy_char(text[inside[j]]):
        j += 1
    for k in reversed(inside[:j]):
        if text[k] in _BOUNDARY_PUNCT:
            ix1 = max(ix1, char_spans[k][1])
            break
    return ix1, ix2


def _promote_formula_like_gaps(ocr_texts, ocr_polys, cell_box, regions):
    """Promote formula-like OCR text gaps to formula regions.

    The layout model occasionally misses small formula fragments (e.g.
    "5m³/h" right after a detected "Q₃=2"). The x intervals of OCR text
    lines not covered by any known formula region are checked, and those
    whose text looks like formula content are returned as extra formula
    regions, trimmed to their formula core so plain text at the gap
    edges stays on the text side.
    """
    x1, y1, x2, y2 = cell_box
    extra = []
    for text, poly in zip(ocr_texts, ocr_polys):
        tx1, ty1, tx2, ty2 = _poly_box(poly)
        cx, cy = _poly_center(poly)
        if not (x1 <= cx < x2 and y1 <= cy < y2):
            continue
        n = len(text)
        w = tx2 - tx1
        if n == 0 or w <= 0:
            continue
        line_h = max(ty2 - ty1, 1.0)
        covered = []
        for rx1, ry1, rx2, ry2 in regions:
            iy1, iy2 = max(ry1, ty1), min(ry2, ty2)
            if iy2 <= iy1 or (iy2 - iy1) / line_h < 0.5:
                continue
            covered.append((max(rx1, tx1), min(rx2, tx2)))
        intervals = []
        cursor = tx1
        for cx1, cx2 in sorted(covered):
            if cx1 > cursor:
                intervals.append((cursor, cx1))
            cursor = max(cursor, cx2)
        if cursor < tx2:
            intervals.append((cursor, tx2))
        char_spans = _char_spans(text, tx1, tx2)
        for ix1, ix2 in intervals:
            s = sum(1 for cx1, cx2 in char_spans if (cx1 + cx2) / 2 < ix1)
            e = sum(1 for cx1, cx2 in char_spans if (cx1 + cx2) / 2 < ix2)
            sub = text[s:e].strip()
            if sub and _looks_like_formula(sub):
                ix1, ix2 = _trim_gap_to_formula_core(
                    text, char_spans, ix1, ix2
                )
                if ix2 > ix1:
                    extra.append([int(ix1), int(ty1), int(ix2), int(ty2)])
    return extra


def _covered_text(text, poly, region):
    """Return (covered_sub_text, covered_fraction) for one OCR line.

    ``covered_sub_text`` is the portion of the OCR text covered by
    ``region`` (None when the region does not overlap the line vertically
    by at least half of the line height), and ``covered_fraction`` is the
    fraction of the line's x range covered by the region.
    """
    tx1, ty1, tx2, ty2 = _poly_box(poly)
    rx1, ry1, rx2, ry2 = region
    iy1, iy2 = max(ry1, ty1), min(ry2, ty2)
    line_h = max(ty2 - ty1, 1.0)
    if iy2 <= iy1 or (iy2 - iy1) / line_h < 0.5:
        return None, 0.0
    ix1, ix2 = max(rx1, tx1), min(rx2, tx2)
    if ix2 <= ix1:
        return None, 0.0
    char_spans = _char_spans(text, tx1, tx2)
    s = sum(1 for cx1, cx2 in char_spans if (cx1 + cx2) / 2 < ix1)
    e = sum(1 for cx1, cx2 in char_spans if (cx1 + cx2) / 2 < ix2)
    return text[s:e], (ix2 - ix1) / max(tx2 - tx1, 1.0)


def _filter_spurious_regions(ocr_texts, ocr_polys, cell_index, regions):
    """Drop formula regions that actually cover plain OCR text.

    The cell layout model sometimes fires on small inline fragments of
    ordinary text (e.g. "6MPa" inside "1.6MPa、1.0MPa"). A region survives
    only when:

    - the OCR text it covers looks like formula content, or
    - it has no meaningful OCR coverage at all (a formula that OCR failed
      to read), or
    - it is much taller than the overlapped text lines (a display formula,
      e.g. with a fraction bar).
    """
    kept = []
    for region in regions:
        pieces = []
        max_line_h = 0.0
        has_coverage = False
        for text, poly in zip(ocr_texts, ocr_polys):
            sub, frac = _covered_text(text, poly, region)
            if sub is None:
                continue
            if frac > 0.05:
                has_coverage = True
                pieces.append(sub)
                box = _poly_box(poly)
                max_line_h = max(max_line_h, box[3] - box[1])
        if not has_coverage:
            kept.append(region)
            continue
        covered_text = "".join(pieces)
        if _looks_like_formula(covered_text):
            kept.append(region)
            continue
        _, ry1, _, ry2 = region
        if max_line_h > 0 and (ry2 - ry1) >= 1.8 * max_line_h:
            kept.append(region)
            continue
        _logger.info(
            "cell %s: dropped spurious formula region %s covering plain text %r",
            cell_index,
            region,
            covered_text,
        )
    return kept


def recognize_table_cells_with_formula(
    table_res,
    doc_img,
    layout_detector,
    formula_predictor,
    text_recognizer=None,
    cell_box_list=None,
    crop_save_dir=None,
    layout_threshold=0.3,
):
    """Recognize formulas inside the cells of one table and fix pred_html.

    Args:
        table_res (dict): one element of the pipeline result's
            ``table_res_list`` (``cell_box_list``, ``pred_html``,
            ``table_ocr_pred``).
        doc_img (np.ndarray): document image after preprocessing
            (``doc_preprocessor_res["output_img"]``); cell coordinates are
            expressed in this image space.
        layout_detector: layout detection model used to analyze each cell
            crop.
        formula_predictor: formula recognition model.
        text_recognizer: text recognition model used to re-recognize the
            plain-text stretches of split lines from the image pixels.
        cell_box_list (list, optional): overrides
            ``table_res["cell_box_list"]`` when given.
        crop_save_dir (str | pathlib.Path, optional): when given, the cell
            and formula crops are saved here for inspection.
        layout_threshold (float): layout detection threshold applied to the
            cell crops.

    Returns:
        list[dict]: per-cell results (JSON-safe dicts with ``cell_index``,
        ``is_formula``, ``formulas``, ``kept_texts``, ...).
    """
    import cv2

    if cell_box_list is None:
        cell_box_list = table_res["cell_box_list"]
    pred_html = table_res["pred_html"]
    ocr_pred = table_res["table_ocr_pred"]
    ocr_texts = ocr_pred["rec_texts"]
    ocr_polys = ocr_pred["rec_polys"]

    td_list = re.findall(r"<td[^>]*>.*?</td>", pred_html, flags=re.S)
    if len(td_list) != len(cell_box_list):
        raise ValueError(
            f"cell count mismatch: {len(td_list)} <td> vs "
            f"{len(cell_box_list)} boxes"
        )

    cell_formula_res = []
    new_tds = []
    for idx, cell_box in enumerate(cell_box_list):
        x1, y1, x2, y2 = [int(v) for v in cell_box]
        crop = doc_img[y1:y2, x1:x2]
        crop_path = None
        if crop_save_dir is not None:
            crop_path = Path(crop_save_dir) / f"cell_{idx}.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_path), crop)
            layout_input = str(crop_path)
        else:
            layout_input = crop

        # Layout analysis on the cell crop: find formula-like sub-regions.
        layout_res = list(
            layout_detector.predict(layout_input, threshold=layout_threshold)
        )
        boxes = layout_res[0]["boxes"]
        labels = [box["label"].lower() for box in boxes]
        formula_boxes = [box for box in boxes if box["label"].lower() == "formula"]

        # Formula regions in doc-image coordinates: deduplicate overlapping
        # detections, then merge fragments of the same formula on one visual
        # line ("Q₃=2" + "5m³/h" in "Q₃=2.5m³/h") so they are recognized
        # together instead of as disjoint fragments.
        # Drop spurious detections first: the cell layout model sometimes
        # fires on plain-text fragments (e.g. "6MPa" in "1.6MPa、1.0MPa"),
        # which would otherwise split surrounding text and duplicate it.
        formula_regions = _filter_spurious_regions(
            ocr_texts,
            ocr_polys,
            idx,
            _merge_adjacent_regions(
                _dedup_regions(
                    [
                        [int(box["coordinate"][0]) + x1,
                         int(box["coordinate"][1]) + y1,
                         int(box["coordinate"][2]) + x1,
                         int(box["coordinate"][3]) + y1]
                        for box in formula_boxes
                    ]
                )
            ),
        )

        # Recover formula fragments missed by the layout model: OCR text
        # gaps on the same line whose content looks like formula text
        # (e.g. ".5m³/h" right after a detected "Q₃=2") are promoted to
        # formula regions and merged, so they are recognized as one complete
        # formula instead of leaking into the plain text output.
        gap_regions = _promote_formula_like_gaps(
            ocr_texts, ocr_polys, (x1, y1, x2, y2), formula_regions
        )
        if gap_regions:
            formula_regions = _merge_adjacent_regions(
                _dedup_regions(formula_regions + gap_regions)
            )
            _logger.info(
                "cell %s: promoted %s formula-like text gap(s) to formula regions",
                idx,
                len(gap_regions),
            )

        if not formula_regions:
            # No formula region: keep the original table recognition result.
            _logger.info("cell %s: no formula region (labels=%s)", idx, labels)
            new_tds.append(td_list[idx])
            cell_formula_res.append(
                {
                    "cell_index": idx,
                    "cell_box": [x1, y1, x2, y2],
                    "crop_path": str(crop_path) if crop_path else None,
                    "layout_labels": labels,
                    "is_formula": False,
                    "content": td_list[idx],
                }
            )
            continue

        # OCR text lines inside this cell. A line overlapping a formula
        # region is split so that only the characters outside the formula
        # regions are kept: the covered characters are replaced by the
        # recognized LaTeX instead of coexisting with the OCR text.
        text_items = []
        cell_ocr_lines = []
        for text, poly in zip(ocr_texts, ocr_polys):
            text_box = _poly_box(poly)
            cx, cy = _poly_center(poly)
            if not (x1 <= cx < x2 and y1 <= cy < y2):
                continue
            cell_ocr_lines.append(text)
            text_area = (text_box[2] - text_box[0]) * (text_box[3] - text_box[1])
            overlap = sum(_intersection_area(text_box, r) for r in formula_regions)
            if text_area <= 0 or overlap / text_area <= 0.05:
                # No significant overlap: keep the whole line.
                text_items.append(
                    (
                        float(cy),
                        float(cx),
                        "text",
                        text,
                        float(text_box[3] - text_box[1]),
                    )
                )
            else:
                # Partial overlap: keep the plain-text stretches outside the
                # formula regions, re-recognized from the image pixels (the
                # line-level OCR text cannot just be cut at the region
                # boundaries, see _split_text_line).
                text_items.extend(
                    _split_text_line(
                        text,
                        text_box,
                        formula_regions,
                        doc_img,
                        text_recognizer,
                    )
                )

        # Recognize each formula region separately (line by line).
        formula_items = []
        for rno, (rx1, ry1, rx2, ry2) in enumerate(formula_regions):
            formula_crop = crop[
                max(0, ry1 - y1):ry2 - y1, max(0, rx1 - x1):rx2 - x1
            ]
            formula_crop_path = None
            if crop_save_dir is not None:
                formula_crop_path = (
                    Path(crop_save_dir) / f"cell_{idx}_formula_{rno}.jpg"
                )
                cv2.imwrite(str(formula_crop_path), formula_crop)
                formula_input = str(formula_crop_path)
            else:
                formula_input = formula_crop

            rec_result = list(formula_predictor.predict(formula_input))
            rec_formula = rec_result[0]["rec_formula"]
            _logger.info("cell %s formula region %s: %s", idx, rno, rec_formula)
            formula_items.append(
                (
                    float((ry1 + ry2) / 2),
                    float((rx1 + rx2) / 2),
                    "formula",
                    rec_formula,
                    float(ry2 - ry1),
                )
            )

        # Rebuild the td content in reading order: group items into visual
        # lines by their vertical centers, then order each line
        # left-to-right and join the lines with <br>.
        line_groups = _group_into_lines(text_items + formula_items)
        td_content = "<br>".join(
            "".join(
                f"$${content}$$" if kind == "formula" else content
                for _, _, kind, content, _ in line
            )
            for line in line_groups
        )
        new_tds.append(f"<td>{td_content}</td>")

        cell_formula_res.append(
            {
                "cell_index": idx,
                "cell_box": [x1, y1, x2, y2],
                "crop_path": str(crop_path) if crop_path else None,
                "layout_labels": labels,
                "is_formula": True,
                "formula_regions": formula_regions,
                "formulas": [f for _, _, _, f, _ in formula_items],
                "kept_texts": [t for _, _, _, t, _ in text_items],
                "cell_ocr_lines": cell_ocr_lines,
                "original_td": td_list[idx],
            }
        )

    td_iter = iter(new_tds)
    fixed_html = re.sub(
        r"<td[^>]*>.*?</td>",
        lambda _: next(td_iter),
        pred_html,
        flags=re.S,
    )
    table_res["pred_html"] = fixed_html
    return cell_formula_res


class PPStructureV3TableCellFormula(PPStructureV3):
    """PP-StructureV3 pipeline with formula recognition inside table cells.

    In addition to the standard PP-StructureV3 behavior, table cell content
    is analyzed with a layout detection model and formula-like sub-regions
    are recognized with a formula recognition model. The table HTML and the
    parsed blocks of the result are updated accordingly.

    Args:
        cell_layout_detection_model_name (str | None): layout detection
            model used to analyze table cell crops. Defaults to
            ``PP-DocLayout_plus-L``.
        cell_layout_detection_model_dir (str | None): model dir for the
            cell layout detection model.
        cell_text_recognition_model_name (str | None): text recognition
            model used to re-recognize plain-text stretches of split lines
            from the image pixels. Defaults to ``PP-OCRv5_server_rec``
            (same as the PP-StructureV3 default table OCR).
        cell_text_recognition_model_dir (str | None): model dir for the
            cell text recognition model.
        cell_formula_layout_threshold (float): layout detection threshold
            applied to the cell crops. Defaults to ``0.3``.
        cell_formula_crop_save_dir (str | None): when given, cell and
            formula crops are saved to this directory for inspection.

    All other arguments are forwarded to :class:`PPStructureV3`. Unless
    ``formula_recognition_model_name`` / ``formula_recognition_model_dir``
    are given, the cell formula recognition uses the same model as the
    pipeline's formula recognition (default ``PP-FormulaNet_plus-M``).
    """

    def __init__(
        self,
        cell_layout_detection_model_name=None,
        cell_layout_detection_model_dir=None,
        cell_text_recognition_model_name=None,
        cell_text_recognition_model_dir=None,
        cell_formula_layout_threshold=None,
        cell_formula_crop_save_dir=None,
        *args,
        **kwargs,
    ):
        self._cell_layout_model_name = (
            cell_layout_detection_model_name or DEFAULT_CELL_LAYOUT_MODEL
        )
        self._cell_layout_model_dir = cell_layout_detection_model_dir
        self._cell_text_rec_model_name = (
            cell_text_recognition_model_name or DEFAULT_CELL_TEXT_REC_MODEL
        )
        self._cell_text_rec_model_dir = cell_text_recognition_model_dir
        self._cell_layout_threshold = (
            0.3 if cell_formula_layout_threshold is None else cell_formula_layout_threshold
        )
        self._cell_crop_save_dir = cell_formula_crop_save_dir
        self._cell_formula_model_name = (
            kwargs.get("formula_recognition_model_name") or DEFAULT_FORMULA_MODEL
        )
        self._cell_formula_model_dir = kwargs.get("formula_recognition_model_dir")
        self._cell_layout_detector = None
        self._cell_formula_predictor = None
        self._cell_text_recognizer = None
        super().__init__(*args, **kwargs)

    def _ensure_cell_models(self):
        if self._cell_layout_detector is None:
            self._cell_layout_detector = create_model(
                self._cell_layout_model_name,
                model_dir=self._cell_layout_model_dir,
                device=self._common_args.get("device"),
                # Disable MKLDNN (run_mode="paddle") to work around the
                # paddle CPU bug: ConvertPirAttribute2RuntimeAttribute not
                # support [pir::ArrayAttribute<pir::DoubleAttribute>].
                engine_config={"run_mode": "paddle"},
            )
        if self._cell_formula_predictor is None:
            self._cell_formula_predictor = FormulaRecognition(
                model_name=self._cell_formula_model_name,
                model_dir=self._cell_formula_model_dir,
                device=self._common_args.get("device"),
            )
        if self._cell_text_recognizer is None:
            self._cell_text_recognizer = TextRecognition(
                model_name=self._cell_text_rec_model_name,
                model_dir=self._cell_text_rec_model_dir,
                device=self._common_args.get("device"),
            )

    def _enhance_table_cells(self, res):
        """Run per-cell formula recognition and fix the result in place."""
        model_settings = res["model_settings"]
        if not model_settings.get("use_table_recognition", False):
            return res
        layout_labels = [
            box["label"].lower() for box in res["layout_det_res"]["boxes"]
        ]
        if "table" not in layout_labels:
            return res
        doc_img = res["doc_preprocessor_res"]["output_img"]
        crop_save_dir = self._cell_crop_save_dir
        for table_res in res["table_res_list"]:
            original_html = table_res["pred_html"]
            cell_formula_res = recognize_table_cells_with_formula(
                table_res,
                doc_img,
                self._cell_layout_detector,
                self._cell_formula_predictor,
                text_recognizer=self._cell_text_recognizer,
                crop_save_dir=crop_save_dir,
                layout_threshold=self._cell_layout_threshold,
            )
            table_res["cell_formula_res"] = cell_formula_res

            # parsing_res_list caches its own copy of the block content
            # (used by markdown/html output), so fix it there as well.
            for block in res["parsing_res_list"]:
                if block.label == "table" and block.content == original_html:
                    block.content = table_res["pred_html"]
                    break
        return res

    def predict_iter(self, input, **kwargs):
        self._ensure_cell_models()
        for res in super().predict_iter(input, **kwargs):
            yield self._enhance_table_cells(res)
