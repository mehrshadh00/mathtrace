# MathTrace

From Pixels to Equations — an experimental image-to-mathematical-art platform.

## Features
- Upload a portrait or capture one with the camera.
- Canny edge extraction and contour tracing.
- Experimental Haar-cascade face-region focus with safe full-image fallback.
- Arc-length resampling and truncated complex Fourier reconstruction.
- SVG, Fourier equation text, and coefficient JSON exports.
- Interactive Desmos graph preview generated directly from the Fourier coefficients.
- Geometric contour-distance diagnostic.

## Run
Create and activate a virtual environment, then install dependencies and launch:

    pip install -r requirements.txt
    streamlit run app.py

## Desmos integration
After tracing an image, choose **Render reconstruction in Desmos** to open an interactive
Desmos Graphing Calculator inside the app. MathTrace converts each complex Fourier
coefficient into a parametric curve and flips the vertical axis to preserve image orientation.

The app uses Desmos' documented demo API key by default for evaluation. For your own
deployment, create a free personal-project key at https://www.desmos.com/my-api and set
`DESMOS_API_KEY` as an environment variable or Streamlit secret.

## Mathematical model
A closed contour is represented by z(t) = x(t) + i y(t), with 0 <= t < 1. Its Fourier expansion is z(t) = sum_k c_k exp(i 2 pi k t). A truncated set of coefficients reconstructs an approximation of the contour.

## Limitations
Face focus is a coarse region-of-interest heuristic, not semantic segmentation. The system does not yet isolate eyes, nose, lips, or hair independently. Its contour-distance metric is geometric, not perceptual.
