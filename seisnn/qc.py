"""
Quality control
"""

import numpy as np


def precision_recall_f1_score(true_positive, pred_count, val_count):
    """
    Calculates precision, recall and f1 score.

    :param int true_positive: True positive count.
    :param int pred_count: Predict count.
    :param int val_count: Validation count.
    :rtype: float
    :return: (precision, recall, f1)
    """
    precision = true_positive / pred_count
    recall = true_positive / val_count
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def signal_to_noise_ratio(signal, noise):
    """
    Calculates power ratio from signal and noise.

    :param numpy.array signal: Signal trace data.
    :param numpy.array noise: Noise trace data.
    :rtype: float
    :return: Signal to noise ratio.
    """
    signal_power = np.sum(np.square(signal))
    noise_power = np.sum(np.square(noise))
    snr = np.log10(signal_power / noise_power)
    return snr


if __name__ == "__main__":
    pass
