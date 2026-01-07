import argparse
import json
import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
from models.bert_classifier import BertClassifier
from attack.textfooler_cn import textfooler_attack

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def inspect_data(path, limit):
    data = load_data(path)
    total = len(data)
    labels = [int(x.get("label", 0)) for x in data]
    pos = sum(1 for l in labels if l == 1)
    neg = sum(1 for l in labels if l == 0)
    lengths = [len(str(x.get("text", ""))) for x in data]
    avg_len = sum(lengths) / total if total > 0 else 0
    print({"count": total, "neg": neg, "pos": pos, "avg_text_len": int(avg_len)})
    for i, x in enumerate(data[:limit]):
        print({"idx": i, "label": int(x.get("label", 0)), "text": x.get("text", "")})

def perturb_text(text, noise_prob, rng):
    chars = list(text)
    out = []
    for ch in chars:
        r = rng.random()
        if r < noise_prob * 0.3:
            continue
        out.append(ch)
        r2 = rng.random()
        if r2 < noise_prob * 0.2:
            out.append(random.choice([" ", ",", ".", "。", "！", "？"]))
    return "".join(out)

def evaluate(model_dir, file, threshold=0.5, perturb=False, noise_prob=0.2, seed=42, pretty=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = BertClassifier(model_dir, device)
    data = load_data(file)
    rng = random.Random(seed)
    texts = [perturb_text(x["text"], noise_prob, rng) if perturb else x["text"] for x in data]
    labels = [int(x["label"]) for x in data]
    probs = np.array([classifier.predict_proba(t) for t in texts])
    preds = (probs[:, 1] >= threshold).astype(int)
    acc = accuracy_score(labels, preds) if len(labels) > 0 else 0.0
    f1m = f1_score(labels, preds, average="macro") if len(labels) > 0 else 0.0
    try:
        ll = float(log_loss(labels, probs, labels=[0,1]))
    except Exception:
        ll = float(0.0)
    avg_conf = float(np.mean([probs[i, preds[i]] for i in range(len(preds))])) if len(preds) > 0 else 0.0
    cm = confusion_matrix(labels, preds, labels=[0,1]) if len(labels) > 0 else np.zeros((2,2), dtype=int)
    tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    success = 0
    total = 0
    for i, sample in enumerate(data):
        text = texts[i]
        label = labels[i]
        if classifier.predict(text) == label:
            total += 1
            adv_text = textfooler_attack(classifier, text, label)
            if classifier.predict(adv_text) != label:
                success += 1
    asr = (success / total) if total > 0 else 0.0
    if pretty:
        print(f"Accuracy: {acc*100:.2f}% | F1: {f1m*100:.2f}% | ASR: {asr*100:.2f}%")
        print(f"Loss: {ll:.4f} | AvgConf: {avg_conf*100:.2f}%")
        print(f"Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    else:
        print({
            "accuracy": round(float(acc)*100, 2),
            "f1_macro": round(float(f1m)*100, 2),
            "log_loss": round(ll, 4),
            "avg_confidence": round(avg_conf*100, 2),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "attack_success_rate": round(float(asr)*100, 2)
        })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./ckpt")
    parser.add_argument("--file", default="data/test.json")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--noise_prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        inspect_data(args.file, args.limit)
    else:
        evaluate(args.model_dir, args.file, threshold=args.threshold, perturb=args.perturb, noise_prob=args.noise_prob, seed=args.seed, pretty=args.pretty)

if __name__ == "__main__":
    main()
