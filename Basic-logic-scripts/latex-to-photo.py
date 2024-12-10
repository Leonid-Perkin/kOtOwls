import matplotlib.pyplot as plt
import io
from PIL import Image

def render_latex_to_image(latex_code, output_path="output.png", dpi=300):
    """
    Render LaTeX code to an image and save it.

    :param latex_code: str, LaTeX code for the formula (e.g., "$\\frac{a}{b}$")
    :param output_path: str, Path to save the rendered image.
    :param dpi: int, Resolution of the output image in dots per inch.
    """
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, latex_code, fontsize=20, ha='center', va='center', 
            transform=ax.transAxes, usetex=True)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    with open(output_path, 'wb') as f:
        f.write(buf.getvalue())
    image = Image.open(buf)
    image.show()
    buf.close()
    plt.close(fig)
latex_code = r"$\begin{array}{c}{{\int x\,d x\,\,+\int x^{\frac{1}{2}}\,d x-3\int x^{5}\,d x+2\int x^{-3}d x\,\,+\int\frac{d x}{S i n^{2}x}+t\,g5\int d x=}}\\ {{\frac{1}{2}x^{2}+\frac{2}{3}\sqrt{x^{3}}-3*\frac{1}{6}x^{6}+2*\frac{1}{(-2)}x^{-2}-(-c\,g t\chi)+t\,g5*x+C}}\end{array}$"
render_latex_to_image(latex_code, output_path="latex_formula.png")