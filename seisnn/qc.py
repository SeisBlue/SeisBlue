import numpy as np


def precision_recall_f1_score(true_positive, pred_count, val_count):
    precision = true_positive / pred_count
    recall = true_positive / val_count
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def signal_to_noise_ratio(signal, noise):
    signal_power = np.sum(np.square(signal))
    noise_power = np.sum(np.square(noise))
    snr = np.log10(signal_power / noise_power)
    return snr
