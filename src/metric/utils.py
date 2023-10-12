import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) != 0:
        return editdistance.eval(target_text, predicted_text) / len(target_text)
    
    if len(predicted_text) != 0:
        return 1

    return 0


def calc_wer(target_text, predicted_text) -> float:
    target_tokens = target_text.split()
    predicted_tokens = predicted_text.split()
    
    if len(target_tokens) != 0:
        return editdistance.eval(target_tokens, predicted_tokens) / len(target_tokens)
    
    if len(predicted_tokens) != 0:
        return 1
    
    return 0
