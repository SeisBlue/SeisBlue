from seisnn.io import get_dir_list
from seisnn.qc import precision_recall_f1_score
from seisnn.plot import plot_error_distribution

predict_pkl_dir = "/mnt/tf_data/pkl/2017_02_predict"
predict_pkl_list = get_dir_list(predict_pkl_dir)

# precision, recall, f1 = precision_recall_f1_score(predict_pkl_list)
# print("Precision = %f, Recall = %f, F1 = %f" % (precision, recall, f1))

plot_error_distribution(predict_pkl_list)
