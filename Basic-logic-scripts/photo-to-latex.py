import os
import torch
from PIL import Image
from pix2tex.cli import LatexOCR
import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("pydantic").setLevel(logging.CRITICAL)
def extract_latex_from_image(image_path):
    """Извлечение формулы в формате LaTeX с изображения с помощью pix2tex."""
    try:
        model = LatexOCR()
        image = Image.open(image_path).convert('RGB')
        latex_formula = model(image)
        return latex_formula
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None

def main():
    print("Приложение для преобразования формулы с фото в LaTeX")
    image_path = input("Введите путь к изображению с формулой: ")
    latex_formula = extract_latex_from_image(image_path)
    if not latex_formula:
        print("Не удалось извлечь формулу из изображения.")
        return
    print(f"Формула в формате LaTeX: {latex_formula}")

if __name__ == "__main__":
    main()
