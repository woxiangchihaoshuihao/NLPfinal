import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification


class BertClassifier:
    def __init__(self, model_path, device):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        ).to(device)
        self.model.eval()

    def predict_proba(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)

        return probs.squeeze().cpu().numpy()

    def predict(self, text):
        return int(self.predict_proba(text).argmax())
