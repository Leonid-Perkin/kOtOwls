import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset as HFDataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
data = {
    "formula_1": [
        r"\frac{d}{dt} \left( \frac{1}{2} m*v^2 \right)",
        r"\frac{d}{dt} (m*v)",
        r"x^2 + y^2 = r^2", 
        r"E = mc^2", 
        r"F = ma",
        r"\int_0^\infty e^{-x^2} dx",
        r"\frac{d}{dx} \left( x^2 \right)",
        r"\frac{a+b}{2}",
        r"y = mx + b",
        r"\sum_{i=1}^{n} i",
        r"\frac{1}{2} mv^2",
        r"\sqrt{x^2 + y^2}",
        r"z = x^2 + y^2",
        r"\log(xy)",
        r"\frac{1}{2} \pi r^2",
    ],
    "formula_2": [
        r"\frac{d}{dt} (m*v)",
        r"\frac{d}{dt} (m*v^2)",
        r"a^2 + b^2 = c^2",
        r"E = mc^2",
        r"F = ma",
        r"\int_{0}^{\infty} e^{-x^2} dx",
        r"\frac{d}{dx} \left( x^2 \right)",
        r"\frac{a+b}{2}",
        r"y = mx + b",
        r"\sum_{i=1}^{n} i",
        r"\frac{1}{2} mv^2",
        r"\sqrt{x^2 + y^2}",
        r"z = x^2 + y^2",
        r"\log(x) + \log(y)",
        r"2 \pi r",
    ],
    "similarity": [
        1,  
        0,  
        2,  
        1, 
        1,  
        1,  
        1,  
        1,  
        1, 
        1,  
        1,  
        2,  
        1,  
        1,  
        2, 
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
        inputs = self.tokenizer(formula_1, formula_2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        return { 
            'input_ids': inputs['input_ids'].squeeze(), 
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(similarity)
        }
dataset = FormulaComparisonDataset(data['formula_1'], data['formula_2'], data['similarity'], tokenizer)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=10,             
    per_device_train_batch_size=8,   
    logging_dir='./logs',            
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
model.save_pretrained("./formula_similarity_model")
tokenizer.save_pretrained("./formula_similarity_model")
