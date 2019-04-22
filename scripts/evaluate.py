from seisnn.io import get_dir_list
from seisnn.pick import get_picks_from_pkl, is_true_positive_pick

validate_pkl_dir = "/mnt/tf_data/pkl/small_set"
validate_pkl_list = get_dir_list(validate_pkl_dir)

predict_pkl_dir = validate_pkl_dir + "_predict"
predict_pkl_list = get_dir_list(predict_pkl_dir)

true_positive = []
val_picks_count = 0
pre_picks_count = 0
for i in range(len(validate_pkl_list)):
    validate_picks = get_picks_from_pkl(validate_pkl_list[i])
    predict_picks = get_picks_from_pkl(predict_pkl_list[i])

    val_picks_count = val_picks_count + len(validate_picks)
    pre_picks_count = pre_picks_count + len(predict_picks)

    for val_pick in validate_picks:
        for pre_pick in predict_picks:
            if is_true_positive_pick(val_pick, pre_pick):
                true_positive.append(val_pick)

precision = len(true_positive) / pre_picks_count
recall = len(true_positive) / val_picks_count
F1 = 2 * (precision * recall) / (precision + recall)

print("precision = %f ,recall = %f ,F1 = %f" % (precision, recall, F1))
