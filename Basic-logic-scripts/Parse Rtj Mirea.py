import os
import sqlite3
from pix2tex import cli
from PIL import Image
from io import BytesIO
import fitz 
DB_NAME = "formulas.db"
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS formulas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    formula_latex TEXT NOT NULL,
    context TEXT NOT NULL
);
''')
conn.commit()
pix2tex_model = cli.LatexOCR()

def process_pdf(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(BytesIO(image_bytes))

                latex_formula = convert_to_latex(image)
                context = f"Page {page_number + 1}, Image {img_index + 1}"

                if latex_formula:
                    save_to_database(latex_formula, context)

    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")

def convert_to_latex(image):
    try:
        latex_formula = pix2tex_model(image)
        return latex_formula
    except Exception as e:
        print(f"Error converting image to LaTeX: {e}")
        return None

def save_to_database(formula, context):
    cursor.execute('''
    INSERT INTO formulas (formula_latex, context) 
    VALUES (?, ?)
    ''', (formula, context))
    conn.commit()

if __name__ == "__main__":
    pdf_path = input("Enter the path to the PDF file: ")
    if os.path.exists(pdf_path):
        process_pdf(pdf_path)
    else:
        print("File does not exist.")
    conn.close()