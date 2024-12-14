import re

def text_to_latex(input_text):
    """
    Преобразует текстовые формулы в формат LaTeX.

    :param input_text: Обычный текст с формулами
    :return: Текст с формулами в формате LaTeX
    """
    try:
        # Используем регулярные выражения для преобразования текста в LaTeX
        transformations = [
            (r'\\b([a-zA-Z])\\^2\\b', r'\1^2'),  # x^2 -> x^2
            (r'\\b([a-zA-Z])_([a-zA-Z0-9]+)\\b', r'\1_\2'),  # x_1 -> x_1
            (r'\\bsqrt\\(([a-zA-Z0-9+\\\-*/^ ]+)\\)', r'\\sqrt{\1}'),  # sqrt(x) -> \sqrt{x}
            (r'\\b([a-zA-Z]+)\\(([a-zA-Z0-9+\\\-*/^ ]+)\\)', r'\1(\2)'),  # sin(x) -> sin(x)
            (r'\\b([0-9]+)\\*([a-zA-Z])\\b', r'\1 \\cdot \2'),  # 2*x -> 2 \cdot x
            (r'\\b([a-zA-Z0-9]+)\\s*\\*\\s*([a-zA-Z0-9]+)\\b', r'\1 \\cdot \2'),  # x * y -> x \cdot y
            (r'([0-9]+)\\^([0-9]+)', r'\1^{\2}'),  # 2^3 -> 2^{3}
        ]

        latex_text = input_text
        for pattern, replacement in transformations:
            latex_text = re.sub(pattern, replacement, latex_text)

        latex_text = f"\\begin{{equation}}\n{latex_text}\n\\end{{equation}}"

        print("Исходная формула:", input_text)
        print("LaTeX формула:", latex_text)
        return latex_text
    except Exception as e:
        print("Ошибка при преобразовании текста в LaTeX:", str(e))
        return None

# Пример использования
if __name__ == "__main__":
    text_formula = "x^2 + y^2 = z^2 and sqrt(x) + 2*y = 10"
    text_to_latex(text_formula)
