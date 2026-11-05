"""Export Fourier contour models to Desmos parametric expressions."""

from __future__ import annotations


def _number(value: float) -> str:
    """Format a finite numeric coefficient for Desmos LaTeX."""
    number = float(value)
    if number == 0:
        return "0"
    return f"{number:.10g}"


def build_desmos_expressions(equations: list[dict]) -> list[dict[str, str]]:
    """Convert complex Fourier coefficients into Desmos parametric curves.

    MathTrace stores z(t) = sum c_k * exp(i * 2*pi*k*t), where c_k = a + bi.
    Thus x(t) = sum(a*cos(theta) - b*sin(theta)) and
    y(t) = sum(a*sin(theta) + b*cos(theta)). Negate y to preserve image
    orientation because pixel coordinates grow downwards.
    """
    expressions: list[dict[str, str]] = []

    for item in equations:
        x_terms: list[str] = []
        y_terms: list[str] = []

        for term in item.get("terms", []):
            frequency = int(term["frequency_index"])
            real = _number(term["real"])
            imag = _number(term["imag"])

            if frequency == 0:
                x_terms.append(real)
                y_terms.append(f"(-({imag}))")
                continue

            angle = f"(2\\pi\\cdot({frequency})t)"
            x_terms.append(f"(({real})\\cos{angle}-({imag})\\sin{angle})")
            y_terms.append(f"(-(({real})\\sin{angle}+({imag})\\cos{angle}))")

        if not x_terms:
            continue

        contour_id = int(item.get("contour", len(expressions)))
        expressions.append(
            {
                "id": f"mathtrace-contour-{contour_id}",
                "latex": (
                    f"\\left({'+'.join(x_terms)},{'+'.join(y_terms)}\\right)"
                    r"\left\{0\le t\le1\right\}"
                ),
            }
        )

    return expressions
