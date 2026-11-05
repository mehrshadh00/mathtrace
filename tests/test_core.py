import numpy as np
from PIL import Image, ImageDraw

from mathtrace.core import equation_text, fourier_reconstruct, render_svg, trace_image

def test_fourier_reconstructs_circle():
    t = np.linspace(0, 2*np.pi, 400, endpoint=False)
    circle = np.column_stack([100+50*np.cos(t), 100+50*np.sin(t)])
    curve, coeffs = fourier_reconstruct(circle, harmonics=8, samples=512)
    assert curve.shape == (512, 2)
    assert np.isfinite(curve).all()
    assert np.count_nonzero(np.abs(coeffs) > 1e-8) <= 17

def test_trace_and_exports():
    image = Image.new("RGB", (240, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((45, 35, 195, 205), outline="black", width=5)
    result = trace_image(image, harmonics=12, max_contours=10)
    assert result.contours
    assert "z(t)" in equation_text(result.equations)
    assert "<svg" in render_svg(result.reconstructed, 240, 240)

def test_face_focus_fallback():
    image = Image.new("RGB", (240, 240), "white")
    ImageDraw.Draw(image).rectangle((30, 30, 210, 210), outline="black", width=5)
    result = trace_image(image, harmonics=8, face_focus=True)
    assert result.face_box is None
    assert result.contours

def test_face_focus_can_be_disabled():
    image = Image.new("RGB", (160, 160), "white")
    ImageDraw.Draw(image).ellipse((25, 20, 135, 140), outline="black", width=4)
    result = trace_image(image, harmonics=8, face_focus=False)
    assert result.face_box is None
    assert result.contours

def test_equation_model_is_complex_fourier():
    points = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
    _, coeffs = fourier_reconstruct(points, harmonics=3)
    assert np.isfinite(coeffs).all()
