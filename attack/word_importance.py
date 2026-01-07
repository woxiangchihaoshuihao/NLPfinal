def word_importance(classifier, text, label):
    words = text.split()
    base_prob = classifier.predict_proba(text)[label]
    importance_scores = []

    for i in range(len(words)):
        new_words = words[:i] + words[i + 1:]
        if len(new_words) == 0:
            importance_scores.append(0.0)
            continue

        new_text = " ".join(new_words)
        new_prob = classifier.predict_proba(new_text)[label]
        importance_scores.append(base_prob - new_prob)

    return importance_scores
