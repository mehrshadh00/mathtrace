"""MathTrace contour extraction and Fourier reconstruction."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

@dataclass
class TraceResult:
    original: Image.Image
    edge_map: Image.Image
    contours: list[np.ndarray]
    reconstructed: list[np.ndarray]
    equations: list[dict]
    mean_error_px: float
    face_box: tuple[int, int, int, int] | None = None
    focus_mode: str = "full-image"

def _prepare_image(image: Image.Image, max_side: int = 1000) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("L")
    image.thumbnail((max_side, max_side))
    return image

def _resample_closed_contour(points: np.ndarray, count: int = 512) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 4:
        return pts
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    if cumulative[-1] <= 1e-9:
        return np.repeat(pts[:1], count, axis=0)
    targets = np.linspace(0, cumulative[-1], count, endpoint=False)
    return np.column_stack([np.interp(targets, cumulative, pts[:, 0]),
                            np.interp(targets, cumulative, pts[:, 1])])

def fourier_reconstruct(points: np.ndarray, harmonics: int = 24,
                        samples: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Fit a closed contour with a truncated complex Fourier series."""
    pts = _resample_closed_contour(points, samples)
    z = pts[:, 0] + 1j * pts[:, 1]
    coeffs = np.fft.fft(z) / len(z)
    order = np.argsort(np.abs(coeffs))[::-1]
    keep = np.zeros(len(coeffs), dtype=bool)
    keep[0] = True
    keep[[int(i) for i in order if i != 0][:max(2, 2 * harmonics)]] = True
    filtered = np.where(keep, coeffs, 0)
    curve = np.fft.ifft(filtered * len(z))
    return np.column_stack([curve.real, curve.imag]), filtered

def _detect_face_region(gray: np.ndarray):
    import cv2
    data = getattr(cv2, "data", None)
    if data is None:
        return None, None
    cascade = cv2.CascadeClassifier(data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        return None, None
    boxes = cascade.detectMultiScale(cv2.equalizeHist(gray), scaleFactor=1.1,
                                     minNeighbors=5, minSize=(48, 48))
    if len(boxes) == 0:
        return None, None
    x, y, w, h = max(boxes, key=lambda b: int(b[2]) * int(b[3]))
    px, py = int(w * .22), int(h * .28)
    left, top = max(0, x-px), max(0, y-py)
    right, bottom = min(gray.shape[1], x+w+px), min(gray.shape[0], y+h+py)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.ellipse(mask, ((left+right)//2, (top+bottom)//2),
                (max(1,(right-left)//2), max(1,(bottom-top)//2)), 0, 0, 360, 255, -1)
    return (int(x), int(y), int(w), int(h)), mask

def _extract_contours(image: Image.Image, min_perimeter: float = 35,
                      face_focus: bool = True):
    import cv2
    gray = np.asarray(image)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 130)
    face_box = None
    if face_focus:
        face_box, mask = _detect_face_region(gray)
        if mask is not None:
            edges = cv2.bitwise_and(edges, edges, mask=mask)
    found, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [c[:, 0, :].astype(float) for c in found
                if cv2.arcLength(c, True) >= min_perimeter and len(c) >= 8]
    contours.sort(key=len, reverse=True)
    return Image.fromarray(edges), contours, face_box

def _symmetric_chamfer(a: np.ndarray, b: np.ndarray, max_points: int = 180) -> float:
    aa, bb = a[::max(1, len(a)//max_points)], b[::max(1, len(b)//max_points)]
    d = np.sqrt(((aa[:, None, :] - bb[None, :, :]) ** 2).sum(axis=2))
    return float((d.min(axis=1).mean() + d.min(axis=0).mean()) / 2)

def trace_image(image: Image.Image, harmonics: int = 24, max_contours: int = 30,
                min_perimeter: float = 35, face_focus: bool = True) -> TraceResult:
    prepared = _prepare_image(image)
    edge_map, contours, face_box = _extract_contours(prepared, min_perimeter, face_focus)
    contours = contours[:max_contours]
    curves, equations, errors = [], [], []
    for index, points in enumerate(contours):
        curve, coeffs = fourier_reconstruct(points, harmonics)
        curves.append(curve)
        errors.append(_symmetric_chamfer(points, curve))
        terms = []
        for k, c in enumerate(coeffs):
            if abs(c) > 1e-5:
                terms.append({"frequency_index": int(k if k <= len(coeffs)//2 else k-len(coeffs)),
                              "real": float(c.real), "imag": float(c.imag),
                              "amplitude": float(abs(c)), "phase": float(np.angle(c))})
        equations.append({"contour": index, "terms": terms})
    mode = "face-focused" if face_box is not None else "full-image (face not detected)"
    return TraceResult(prepared, edge_map, contours, curves, equations,
                       float(np.mean(errors)) if errors else 0.0, face_box, mode)

def render_svg(curves: Iterable[np.ndarray], width: int, height: int,
               stroke_width: float = 1.6) -> str:
    paths = []
    for curve in curves:
        if len(curve) < 2:
            continue
        commands = [f"M {curve[0,0]:.2f} {curve[0,1]:.2f}"]
        commands.extend(f"L {x:.2f} {y:.2f}" for x, y in curve[1:])
        commands.append("Z")
        paths.append(f'<path d="{" ".join(commands)}" fill="none" stroke="#111" '
                     f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
            + "".join(paths) + "</svg>")

def equation_text(equations: list[dict]) -> str:
    lines = ["# MathTrace Fourier contour model",
             "# z(t) = sum_k c_k * exp(i * 2*pi*k*t), 0 <= t < 1",
             "# x(t) = Re(z(t)); y(t) = Im(z(t))", ""]
    for item in equations:
        lines.append(f"## Contour {item['contour']}")
        for term in item["terms"]:
            lines.append("c_k = {real:.8f} {imag:+.8f}i (k={frequency_index}, "
                         "amplitude={amplitude:.6f}, phase={phase:.6f})".format(**term))
        lines.append("")
    return "\n".join(lines)
