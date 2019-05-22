from seisnn.io import get_dir_list
from seisnn.qc import precision_recall_f1_score
from seisnn.plot import plot_error_distribution

predict_dataset_dir = "/mnt/tf_data/dataset/2018_02_18_predict"
predict_dataset_list = get_dir_list(predict_dataset_dir)

precision, recall, f1 = precision_recall_f1_score(predict_dataset_list)
print("Precision = %f, Recall = %f, F1 = %f" % (precision, recall, f1))

plot_error_distribution(predict_dataset_list)
