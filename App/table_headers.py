"""Start receipt contents below the first header confirmed by two columns.

Supports Kazakh/Russian labels, including numeric OCR of the number sign (N8).
Without a confirmed header no upper content is discarded by this boundary.
"""
import re
import unicodedata


def without_table_headers(lines, layout):
    def role(line):
        text = unicodedata.normalize("NFKC", line["text"]).casefold().replace("ғ", "г")
        key = re.sub(r"[^a-zа-яәіңүұқөһ]", "", text)
        x = line["center"][0]
        if x < layout["name_right"] and any(word in key for word in ("атау", "атайы", "наименование", "название")):
            return "name"
        if layout["name_right"] <= x < layout["quantity_right"] and key in {"саны", "колво", "количество"}:
            return "quantity"
        if layout["quantity_right"] <= x < layout["right"] + 4 and key in {"багасы", "багас", "цена", "стоимость"}:
            return "price"
        return None

    def aligned(first, second):
        heights = []
        for line in (first, second):
            box = line.get("eval_box", line.get("box", []))
            ys = [float(point[1]) for point in box]
            heights.append(max(1., max(ys) - min(ys)) if ys else 1.)
        return (max(heights) <= 3 * min(heights) and
                abs(first["center"][1] - second["center"][1]) <= max(6., sum(heights) * .3))

    labels = [(line, role(line)) for line in lines if role(line)]
    ignored = set()
    header_bottoms = []
    for anchor, _ in labels:
        cohort = [(line, label) for line, label in labels if aligned(anchor, line)]
        # Require evidence from two different columns, rather than a product name.
        if len({label for _, label in cohort}) < 2:
            continue
        header_bottoms.append(max(
            max((float(point[1]) for point in line.get('eval_box', line.get('box', []))),
                default=float(line['center'][1])) for line, _ in cohort))
        for line in lines:
            if aligned(anchor, line) and (role(line) or not re.search(r"\d", line["text"])):
                ignored.add(id(line))
    first_header_bottom = min(header_bottoms) if header_bottoms else None
    return [line for line in lines if id(line) not in ignored and
            (first_header_bottom is None or float(line['center'][1]) > first_header_bottom)]
