from sklearn.metrics import f1_score
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

def calculate_context_recall(sources, answer):
    match_score = max([SequenceMatcher(None, src, answer).ratio() for src in sources])
    return match_score

def calculate_exact_match(answer, query):
    return int(query.lower().strip("?") in answer.lower())

def calculate_f1(prediction, reference):
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    common = set(pred_tokens) & set(ref_tokens)

    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_bleu(prediction, reference):
    reference_tokens = [reference.lower().split()]
    prediction_tokens = prediction.lower().split()
    return sentence_bleu(reference_tokens, prediction_tokens)

def calculate_rouge(prediction, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(reference, prediction)['rougeL'].fmeasure
