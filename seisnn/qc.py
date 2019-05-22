from seisnn.pick import get_picks_from_dataset


def precision_recall_f1_score(predict_pkl_list):
    true_positive = 0
    val_picks_count = 0
    pre_picks_count = 0
    for i, pkl in enumerate(predict_pkl_list):
        picks = get_picks_from_dataset(pkl)
        for p in picks:
            if p.evaluation_mode == "manual":
                val_picks_count += 1

            elif p.evaluation_mode == "automatic":
                pre_picks_count += 1
                if p.evaluation_status == "confirmed":
                    true_positive += 1

        if i % 1000 == 0:
            print("Reading... %d out of %d " % (i, len(predict_pkl_list)))

    precision = true_positive / pre_picks_count
    recall = true_positive / val_picks_count
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1
