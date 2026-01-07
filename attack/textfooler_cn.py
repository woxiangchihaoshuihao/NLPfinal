from attack.word_importance import word_importance
from attack.synonym import get_synonyms
import copy


def replace_word(words, index, new_word):
    new_words = copy.deepcopy(words)
    new_words[index] = new_word
    return " ".join(new_words)


def textfooler_attack(classifier, text, true_label):
    words = text.split()

    importance = word_importance(classifier, text, true_label)
    sorted_indices = sorted(
        range(len(importance)),
        key=lambda i: importance[i],
        reverse=True
    )

    original_pred = classifier.predict(text)

    for idx in sorted_indices:
        word = words[idx]
        synonyms = get_synonyms(word)

        for syn in synonyms:
            adv_text = replace_word(words, idx, syn)
            adv_pred = classifier.predict(adv_text)

            if adv_pred != original_pred:
                return adv_text

    return text
