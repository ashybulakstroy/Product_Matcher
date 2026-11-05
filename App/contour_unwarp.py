"""Generate reviewable nonlinear rectifications from saved table polygons.

Run explicitly with python -m App.contour_unwarp. This does not change the
conservative automatic OCR selector. All sampling uses the original pixels once.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _save_png(path: Path, image: np.ndarray):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Unable to encode {path}")
    data = encoded.tobytes()
    # Image viewers on Windows can hold files open. Identical reruns need no
    # write access and should not disturb an image currently being inspected.
    if path.exists() and path.read_bytes() == data:
        return
    path.write_bytes(data)


def _fit(x, y, degree):
    """Robust low-degree fit; isolated contour detours must not bend the page."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    degree = min(degree, len(np.unique(x)) - 1)
    weights = np.ones(len(x))
    for _ in range(8):
        fit = np.polynomial.polynomial.polyfit(x, y, degree, w=weights)
        residual = y - np.polynomial.polynomial.polyval(x, fit)
        sigma = max(.004, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        weights = np.minimum(1., 1.5 * sigma / np.maximum(np.abs(residual), 1e-6))
    return fit


def _supported_top_right(points, top_left, estimated, origin, extent):
    """Return a real top-rule vertex when the fitted right side overshoots it.

    A stepped total cell can pull the fit of the right outer side to the right.
    Only a contour vertex on the printed top rule may correct that extrapolation;
    the other three body corners remain untouched.
    """
    physical = points * extent + origin
    tl, tr = top_left, estimated
    body_height = max(1., float(np.linalg.norm(physical[3] - tl)))
    candidates = points[(physical[:, 0] > (tl[0] + tr[0]) / 2)
                        & (physical[:, 1] <= min(tl[1], tr[1]) + max(12., body_height * .04))
                        & (physical[:, 0] <= tr[0] + 8.)]
    if not len(candidates):
        return None
    # A contour may contain header-cell intersections on the top line.  The
    # true top-right table corner is the outermost (rightmost) supported one,
    # not the point nearest to an estimate pulled by a lower footer step.
    candidate = max(candidates, key=lambda point: float(point[0]))
    return candidate if tr[0] - (candidate[0] * extent[0] + origin[0]) >= 16. else None


def build_contour_map(polygon: np.ndarray, margin: int = 16, trust_top: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """Coons patch bounded by fitted top, body bottom and two side curves.

    The total-cell step determines the extra strip below the table body; it is
    never stretched over the entire table. Small text spurs are not control points.
    """
    points = np.asarray(polygon, np.float64).reshape(-1, 2)
    if len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("A finite table polygon with at least four points is required")
    origin = points.min(axis=0)
    extent = np.ptp(points, axis=0)
    if min(extent) < 10:
        raise ValueError("Degenerate table boundary")
    pts = (points - origin) / extent
    x, y = pts.T
    # Only the outer side bands are eligible. A text spur or a footer corner
    # inside the table must not become a side control point.
    side_fits = []
    for left in (True, False):
        select = (x < .1 if left else x > .9)
        yy, xx = y[select], x[select]
        # Several footer/text vertices can share almost the same height. Only
        # the outer envelope belongs to the side of the table.
        order = np.argsort(yy)
        groups = []
        for index in order:
            if not groups or yy[index] - yy[groups[-1][0]] > .06:
                groups.append([index])
            else:
                groups[-1].append(index)
        envelope = [min(group, key=lambda i: xx[i]) if left else max(group, key=lambda i: xx[i]) for group in groups]
        yy, xx = yy[envelope], xx[envelope]
        fit = _fit(yy, xx, 1)
        if len(yy) >= 5 and np.ptp(yy) > .7:
            quadratic = _fit(yy, xx, 2)
            if abs(quadratic[2]) < .1:
                fit = quadratic
        side_fits.append(fit)
    top_select = y < (.16 if trust_top else .04)
    # Fit a bent top only if there are points across the width. Otherwise use
    # the supported segment and extrapolate across a missing first/last corner.
    top_fit = _fit(x[top_select], y[top_select], 1)
    top_x = np.unique(x[top_select])
    top_y = np.array([np.min(y[top_select & (x == value)]) for value in top_x])

    def top_curve(values):
        if not trust_top or len(top_x) < 2:
            return np.polynomial.polynomial.polyval(values, top_fit)
        result = np.interp(values, top_x, top_y)
        left_slope = right_slope = top_fit[1] if len(top_fit) > 1 else 0.
        return np.where(values < top_x[0], top_y[0] + (values-top_x[0])*left_slope,
                        np.where(values > top_x[-1], top_y[-1]+(values-top_x[-1])*right_slope, result))

    # Long lower-left edges belong to the body; lower-right edges to the total.
    edges = []
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        a, b = (a, b) if a[0] <= b[0] else (b, a)
        dx, dy = b - a
        slope_pixels = abs(dy * extent[1]) / max(dx * extent[0], 1e-6)
        if (dx > .18 and min(a[1], b[1]) > .65 and slope_pixels < .35
                and not (a[0] > .3 and b[0] > .85)):
            edges.append((dx, a, b))
    if not edges:
        raise ValueError("No supported lower table-body boundary")
    _, a, b = max(edges, key=lambda edge: edge[0])
    bottom_fit = _fit([a[0], b[0]], [a[1], b[1]], 1)

    def curve(c, values):
        return np.polynomial.polynomial.polyval(values, c)

    corners = []
    for side, horizontal, initial_y in ((side_fits[0], None, 0),
                                         (side_fits[1], None, 0),
                                         (side_fits[1], bottom_fit, 1),
                                         (side_fits[0], bottom_fit, 1)):
        cy = initial_y
        for _ in range(12):
            cx = curve(side, cy)
            cy = top_curve(cx) if horizontal is None else curve(horizontal, cx)
        corners.append([cx, cy])
    tl, tr, br, bl = np.asarray(corners)
    estimated_top_right = (tr * extent + origin).copy()
    supported_top_right = _supported_top_right(pts, tl * extent + origin, estimated_top_right, origin, extent)
    refinement = dict(applied=False)
    right_top_shift = 0.
    if supported_top_right is not None:
        right_top_shift = float(supported_top_right[0] - tr[0])
        tr[0] = supported_top_right[0]
        tr[1] = top_curve(tr[0])
        refinement = dict(applied=True, estimated=estimated_top_right.tolist(),
                          supported=(supported_top_right * extent + origin).tolist(),
                          shift_px=float(right_top_shift * extent[0]))
    physical = np.asarray(corners) * extent + origin
    width = max(16, int(round((np.linalg.norm(physical[1]-physical[0])
                              + np.linalg.norm(physical[2]-physical[3])) / 2)))
    height = max(16, int(round((np.linalg.norm(physical[3]-physical[0])
                               + np.linalg.norm(physical[2]-physical[1])) / 2)))
    body_fraction = ((bl[1] - tl[1]) + (br[1] - tr[1])) / 2
    if body_fraction <= .4:
        raise ValueError("Implausible table-body height")
    extra_fraction = max(0., float(np.max(y - curve(bottom_fit, x)))) / body_fraction
    footer_height = int(np.ceil(extra_fraction * height))
    if footer_height > height * .25:
        raise ValueError("Implausibly large total extension")
    if max(width, height + footer_height) > 10000:
        raise ValueError("Unsafe output size")

    u = ((np.arange(width + 2*margin, dtype=np.float64) - margin) / max(width-1, 1))[None, :]
    v = ((np.arange(height + footer_height + 2*margin, dtype=np.float64) - margin) / max(height-1, 1))[:, None]

    def patch(u, v):
        tx = tl[0] + (tr[0]-tl[0])*u
        bx = bl[0] + (br[0]-bl[0])*u
        ly = tl[1] + (bl[1]-tl[1])*v
        ry = tr[1] + (br[1]-tr[1])*v
        top = [tx, top_curve(tx)]
        bottom = [bx, curve(bottom_fit, bx)]
        left = [curve(side_fits[0], ly), ly]
        # Anchor only the top end of the right side to the printed corner.
        # The correction fades to zero at the body bottom, preserving the
        # separately detected total-cell step and all lower geometry.
        right = [curve(side_fits[1], ry) + right_top_shift * (1 - np.clip(ry, 0, 1)), ry]
        values = []
        for axis in range(2):
            bilinear = ((1-u)*(1-v)*tl[axis] + u*(1-v)*tr[axis]
                        + u*v*br[axis] + (1-u)*v*bl[axis])
            values.append((1-v)*top[axis] + v*bottom[axis]
                          + (1-u)*left[axis] + u*right[axis] - bilinear)
        return values

    # Continue below/above the body using its boundary derivative. Polynomial
    # extrapolation outside [0,1] could distort the total or adjacent text.
    clipped_v = np.clip(v, 0, 1)
    values = patch(u, clipped_v)
    derivative = [(b-a)/.01 for a, b in zip(patch(u, np.maximum(0, clipped_v-.01)),
                                           patch(u, np.minimum(1, clipped_v+.01)))]
    # At 0/1 the finite difference is one-sided (interval .01).
    for axis in range(2):
        values[axis] += (v-clipped_v)*derivative[axis]
    map_x = (values[0]*extent[0] + origin[0]).astype(np.float32)
    map_y = (values[1]*extent[1] + origin[1]).astype(np.float32)
    dx_y, dx_x = np.gradient(map_x)
    dy_y, dy_x = np.gradient(map_y)
    jacobian = dx_x*dy_y - dx_y*dy_x
    if not np.isfinite(jacobian).all() or np.min(jacobian) <= .02:
        raise ValueError("Nonlinear map folds or collapses pixels")
    metadata = dict(method="contour_coons_patch", algorithm_version="contour-unwarp-1.0",
                    source_polygon=points.tolist(), source_body_corners=physical.tolist(),
                    body_size=[width, height], footer_height=footer_height, margin=margin,
                    output_size=[map_x.shape[1], map_x.shape[0]],
                    min_jacobian=float(jacobian.min()), max_jacobian=float(jacobian.max()),
                    fitted_top=top_fit.tolist(), fitted_bottom=bottom_fit.tolist(),
                    top_knots=np.column_stack([top_x, top_y]).tolist(), trusted_top=trust_top,
                    fitted_left=side_fits[0].tolist(), fitted_right=side_fits[1].tolist(),
                    normalization_origin=origin.tolist(), normalization_extent=extent.tolist(),
                    top_right_refinement=refinement)
    return map_x, map_y, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contours", type=Path, help="Optional saved boundary JSON; otherwise detect input photos")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval_outputs")
    args = parser.parse_args()
    if args.contours:
        contours = json.loads(args.contours.read_text(encoding="utf-8-sig"))
    else:
        from App.table_boundary import detect_table_boundary
        contours = []
        for source in sorted((ROOT / "data/in_data").glob("*.jpeg")):
            boundary = detect_table_boundary(cv2.imread(str(source)))
            if boundary["polygon"] is not None:
                contours.append(dict(file=source.name,polygon=boundary["polygon"].tolist(),
                                     status=boundary["status"],warnings=boundary["warnings"]))
    summary, cards, tiles = [], [], []
    for row in contours:
        path = ROOT / "data/in_data" / row["file"]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode {path}")
        directory = args.out_dir / path.stem
        directory.mkdir(parents=True, exist_ok=True)
        try:
            map_x, map_y, metadata = build_contour_map(row["polygon"], trust_top=row["status"] == "detected")
            output = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))
            _save_png(directory / "rectified.png", output)
            # Compatibility with the existing per-photo review filenames.
            _save_png(directory / "warped.png", output)
            np.savez_compressed(directory / "rectification_map.npz", map_x=map_x, map_y=map_y)
            metadata.update(file=path.name, output_file="rectified.png", status="review_preview", boundary_status=row["status"],
                            warnings=row["warnings"], original_size=[image.shape[1], image.shape[0]])
            (directory / "rectification.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            summary.append(metadata)
            link = f"{path.stem}/rectified.png"
            cards.append(f'<article><h2>{html.escape(path.stem)}</h2><p>{html.escape(", ".join(row["warnings"]))}</p>'
                         f'<a href="{html.escape(link)}"><img src="{html.escape(link)}"></a></article>')
            tile = np.full((560, 340, 3), 225, np.uint8)
            factor = min(330/output.shape[1], 520/output.shape[0])
            small = cv2.resize(output, None, fx=factor, fy=factor)
            tile[:small.shape[0], :small.shape[1]] = small
            label = f'{len(tiles)}: {path.stem.removeprefix("WhatsApp Image ")}'
            cv2.putText(tile, label, (3, 544), 0, .42, (0, 0, 0), 1)
            tiles.append(tile)
            print(f"OK {path.name}: {output.shape[1]}x{output.shape[0]}", flush=True)
        except ValueError as error:
            summary.append(dict(file=path.name, status="failed", error=str(error)))
            print(f"FAILED {path.name}: {error}", flush=True)
    (args.out_dir / "rectification_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if tiles:
        blank = np.full_like(tiles[0], 225)
        montage = np.vstack([np.hstack(tiles[i:i+4]+[blank]*(4-len(tiles[i:i+4]))) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(args.out_dir / "rectified_contact.jpg"), montage)
    (args.out_dir / "rectified_review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Выпрямленные таблицы</title>'
        '<style>body{font:16px system-ui;background:#ddd;margin:24px}main{display:grid;grid-template-columns:'
        'repeat(auto-fit,minmax(350px,1fr));gap:24px}article{background:white;padding:16px}'
        'h2{font-size:15px}img{width:100%;height:auto}p{color:#a30}</style>'
        '<h1>Выпрямленные таблицы</h1><p>Варианты для визуальной проверки. '
        'Основная таблица и отдельная нижняя ячейка итога. Нажмите для полного разрешения.</p><main>'
        + ''.join(cards) + '</main>', encoding="utf-8")


if __name__ == "__main__":
    main()
