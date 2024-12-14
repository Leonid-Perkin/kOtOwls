import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset as HFDataset

# Загружаем токенизатор и модель (предполагаем, что модель уже обучена для текста)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)  # 3 метки: схожесть 0, 1, 2

# Пример данных
data = {
    "formula_1": [
        r"\frac{d}{dt} \left( \frac{1}{2} m*v^2 \right)",  # Формула 1
        r"\frac{d}{dt} (m*v)",  # Формула 2
        r"x^2 + y^2 = r^2",  # Формула 3
        r"E = mc^2",  # Формула 4
        r"F = ma",  # Формула 5
        r"\int_0^\infty e^{-x^2} dx",  # Формула 6
        r"\frac{d}{dx} \left( x^2 \right)",  # Формула 7
        r"\frac{a+b}{2}",  # Формула 8
        r"y = mx + b",  # Формула 9
        r"\sum_{i=1}^{n} i",  # Формула 10
        r"\frac{1}{2} mv^2",  # Формула 11
        r"\sqrt{x^2 + y^2}",  # Формула 12
        r"z = x^2 + y^2",  # Формула 13
        r"\log(xy)",  # Формула 14
        r"\frac{1}{2} \pi r^2",  # Формула 15
    ],
    "formula_2": [
        r"\frac{d}{dt} (m*v)",  # Формула 1
        r"\frac{d}{dt} (m*v^2)",  # Формула 2
        r"a^2 + b^2 = c^2",  # Формула 3
        r"E = mc^2",  # Формула 4
        r"F = ma",  # Формула 5
        r"\int_{0}^{\infty} e^{-x^2} dx",  # Формула 6
        r"\frac{d}{dx} \left( x^2 \right)",  # Формула 7
        r"\frac{a+b}{2}",  # Формула 8
        r"y = mx + b",  # Формула 9
        r"\sum_{i=1}^{n} i",  # Формула 10
        r"\frac{1}{2} mv^2",  # Формула 11
        r"\sqrt{x^2 + y^2}",  # Формула 12
        r"z = x^2 + y^2",  # Формула 13
        r"\log(x) + \log(y)",  # Формула 14
        r"2 \pi r",  # Формула 15
    ],
    "similarity": [
        1,  # Формулы 1
        0,  # Формулы 2
        2,  # Формулы 3
        1,  # Формулы 4
        1,  # Формулы 5
        1,  # Формулы 6
        1,  # Формулы 7
        1,  # Формулы 8
        1,  # Формулы 9
        1,  # Формулы 10
        1,  # Формулы 11
        2,  # Формулы 12
        1,  # Формулы 13
        1,  # Формулы 14
        2,  # Формулы 15
    ]
}


# Создание датасета
class FormulaComparisonDataset(Dataset):
    def __init__(self, formulas_1, formulas_2, similarities, tokenizer, max_length=128):
        self.formulas_1 = formulas_1
        self.formulas_2 = formulas_2
        self.similarities = similarities
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.formulas_1)

    def __getitem__(self, idx):
        formula_1 = self.formulas_1[idx]
        formula_2 = self.formulas_2[idx]
        similarity = self.similarities[idx]
        
        # Токенизация двух формул
        inputs = self.tokenizer(formula_1, formula_2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        
        # Возвращаем токенизированные данные и метку сходства
        return { 
            'input_ids': inputs['input_ids'].squeeze(), 
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(similarity)
        }

# Инициализация датасета и DataLoader
dataset = FormulaComparisonDataset(data['formula_1'], data['formula_2'], data['similarity'], tokenizer)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Настройки обучения
training_args = TrainingArguments(
    output_dir='./results',          # где сохранять результаты
    num_train_epochs=10,              # количество эпох
    per_device_train_batch_size=8,   # размер батча
    logging_dir='./logs',            # директория для логов
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Запуск обучения
trainer.train()

# Сохранение модели
model.save_pretrained("./formula_similarity_model")
tokenizer.save_pretrained("./formula_similarity_model")
