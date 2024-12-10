from sympy import symbols, latex, parse_expr
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application
import re

def preprocess_expression(expression):
    """
    Предобрабатывает выражение, исправляя общие ошибки и приводя функции к стандартному виду.

    :param expression: Строка с математическим выражением
    :return: Исправленная строка выражения
    """
    replacements = {
        r'\btg\b': 'tan',  # Замена tg -> tan
        r'\bctg\b': 'cot', # Замена ctg -> cot
        r'\bsh\b': 'sinh', # Замена sh -> sinh
        r'\bch\b': 'cosh', # Замена ch -> cosh
        r'\bth\b': 'tanh', # Замена th -> tanh
    }

    for pattern, replacement in replacements.items():
        expression = re.sub(pattern, replacement, expression)
    expression = expression.replace('^', '**').strip()
    return expression

def to_latex(expression):
    """
    Преобразует текстовое выражение в LaTeX-формат.

    :param expression: Строка с математическим выражением
    :return: Строка в формате LaTeX
    """
    transformations = (standard_transformations + (implicit_multiplication_application,))
    try:
        expression = preprocess_expression(expression)
        parsed_expr = parse_expr(expression, transformations=transformations)
        latex_expr = latex(parsed_expr)
        return latex_expr
    except Exception as e:
        return f"Ошибка при обработке выражения: {e}"
expression = "x + sqrt(x) - 3*x^5 + 2/x^3 - 1/sin(x)^2 + tg(5)"
latex_output = to_latex(expression)
print("LaTeX формула:", latex_output)
