"""
Quality control
"""

import numpy as np


def precision_recall_f1_score(true_positive, pred_count, val_count):
    """
    Calculates precision, recall and f1 score.

    :type true_positive: int
    :param true_positive: True positive count.
    :type pred_count: int
    :param pred_count: Predict count.
    :type val_count: int
    :param val_count: Validation count.
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

    :type signal: numpy.array
    :param signal: Signal trace data.
    :type noise: numpy.array
    :param noise: Noise trace data.
    :rtype: float
    :return: Signal to noise ratio.
    """
    signal_power = np.sum(np.square(signal))
    noise_power = np.sum(np.square(noise))
    snr = np.log10(signal_power / noise_power)
    return snr


if __name__ == "__main__":
    pass
