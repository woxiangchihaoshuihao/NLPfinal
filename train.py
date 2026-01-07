import json
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-chinese"

tokenizer = BertTokenizerFast.from_pretrained(model_name)


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class FraudDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.texts = [x["text"] for x in data]
        self.labels = [x["label"] for x in data]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        item = {k: v.squeeze() for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


train_data = load_data("data/train.json")
train_dataset = FraudDataset(train_data)

model = BertForSequenceClassification.from_pretrained(
    model_name, num_labels=2
).to(device)

args = TrainingArguments(
    output_dir="./ckpt",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_steps=50,
    save_steps=500
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset
)

trainer.train()
trainer.save_model("./ckpt")
tokenizer.save_pretrained("./ckpt")
