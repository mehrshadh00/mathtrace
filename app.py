import json

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from mathtrace.core import equation_text, render_svg, trace_image

st.set_page_config(page_title="MathTrace", page_icon="∑", layout="wide")
st.title("∑ MathTrace")
st.caption("From pixels to equations — reconstruct image contours with Fourier mathematics.")
with st.sidebar:
    st.header("Reconstruction controls")
    harmonics = st.slider("Fourier complexity (harmonics)", 4, 100, 28, 2)
    max_contours = st.slider("Maximum contours", 1, 100, 35)
    min_perimeter = st.slider("Ignore contours shorter than (px)", 10, 180, 35, 5)
    face_focus = st.toggle("Face-aware focus (experimental)", value=True)
    st.info("Face focus uses OpenCV Haar detection and falls back to full-image tracing.")
left, right = st.columns(2)
with left:
    uploaded = st.file_uploader("Upload a portrait or image", type=["png", "jpg", "jpeg", "webp"])
with right:
    camera_photo = st.camera_input("Or take a photo")
if uploaded:
    source = Image.open(uploaded)
elif camera_photo:
    source = Image.open(camera_photo)
else:
    st.info("Upload an image or take a photo to begin.")
    st.stop()
if st.button("Trace image into equations", type="primary"):
    with st.spinner("Extracting contours and computing Fourier descriptors..."):
        st.session_state["result"] = trace_image(source, harmonics, max_contours,
                                                 min_perimeter, face_focus)
result = st.session_state.get("result")
if result:
    c1, c2 = st.columns(2)
    c1.image(result.edge_map, caption="Extracted edges", use_container_width=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    for curve in result.reconstructed:
        ax.plot(curve[:, 0], curve[:, 1], color="black", linewidth=.85)
    ax.set_xlim(0, result.original.width)
    ax.set_ylim(result.original.height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    c2.pyplot(fig, clear_figure=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Contours modeled", len(result.contours))
    m2.metric("Harmonics / contour", harmonics)
    m3.metric("Mean contour distance", f"{result.mean_error_px:.2f} px")
    st.caption(f"Processing mode: {result.focus_mode}")
    st.download_button("Download SVG", render_svg(result.reconstructed, result.original.width,
                         result.original.height), "mathtrace-reconstruction.svg", "image/svg+xml")
    txt = equation_text(result.equations)
    st.download_button("Download equations", txt, "mathtrace-equations.txt", "text/plain")
    st.download_button("Download coefficient JSON", json.dumps(result.equations, indent=2),
                       "mathtrace-coefficients.json", "application/json")
    with st.expander("Inspect equations"):
        st.code(txt[:18000], language="text")
