import html
import json
import os

import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from mathtrace.core import equation_text, render_svg, trace_image
from mathtrace.desmos import build_desmos_expressions


def _desmos_embed(expressions: list[dict], width: int, height: int) -> str:
    """Build an isolated HTML document containing an interactive Desmos graph."""
    # The documented public demo key keeps the feature usable out of the box.
    # Set DESMOS_API_KEY (or Streamlit secret) to use a personal API key.
    api_key = os.getenv("DESMOS_API_KEY")
    try:
        api_key = api_key or st.secrets.get("DESMOS_API_KEY")
    except Exception:
        # Streamlit raises when no secrets file is configured.
        pass
    api_key = api_key or "dcb31709b452b1cf9dc26972add0fda6"

    safe_key = html.escape(str(api_key), quote=True)
    expression_json = json.dumps(expressions).replace("<", "\\u003c")
    return f"""
    <!doctype html>
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
          html, body, #calculator {{ margin: 0; width: 100%; height: 100%; }}
          body {{ overflow: hidden; }}
        </style>
        <script src="https://www.desmos.com/api/v1.11/calculator.js?apiKey={safe_key}"></script>
      </head>
      <body>
        <div id="calculator"></div>
        <script>
          const calculator = Desmos.GraphingCalculator(
            document.getElementById("calculator"),
            {{ expressions: true, keypad: true, settingsMenu: true, zoomButtons: true }}
          );
          calculator.setExpressions({expression_json});
          calculator.setMathBounds({{
            left: 0,
            right: {int(width)},
            bottom: -{int(height)},
            top: 0
          }});
        </script>
      </body>
    </html>
    """


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
        st.session_state["result"] = trace_image(
            source, harmonics, max_contours, min_perimeter, face_focus
        )
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
    st.download_button(
        "Download SVG",
        render_svg(result.reconstructed, result.original.width, result.original.height),
        "mathtrace-reconstruction.svg",
        "image/svg+xml",
    )
    txt = equation_text(result.equations)
    st.download_button("Download equations", txt, "mathtrace-equations.txt", "text/plain")
    st.download_button(
        "Download coefficient JSON",
        json.dumps(result.equations, indent=2),
        "mathtrace-coefficients.json",
        "application/json",
    )

    st.subheader("Interactive Desmos graph")
    desmos_expressions = build_desmos_expressions(result.equations)
    if desmos_expressions:
        st.caption(
            f"Plotting {len(desmos_expressions)} Fourier contour(s) in Desmos. "
            "The graph keeps the original image orientation."
        )
        if st.button("Render reconstruction in Desmos", type="primary"):
            st.session_state["show_desmos"] = True
        if st.session_state.get("show_desmos"):
            components.html(
                _desmos_embed(
                    desmos_expressions, result.original.width, result.original.height
                ),
                height=700,
                scrolling=False,
            )
    else:
        st.info("No contour equations are available to plot.")

    with st.expander("Inspect equations"):
        st.code(txt[:18000], language="text")
