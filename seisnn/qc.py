def precision_recall_f1_score(true_positive, pred_count, val_count):
    precision = true_positive / pred_count
    recall = true_positive / val_count
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1
