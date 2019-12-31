import os
import argparse
import pandas as pd

from seisnn.qc import precision_recall_f1_score
from seisnn.plot import plot_error_distribution
from seisnn.core import Feature
from seisnn.utils import get_config
from seisnn.io import read_dataset
from seisnn.pick import validate_picks_nearby, get_time_residual

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=False, help='dataset', type=str)
args = ap.parse_args()

config = get_config()
dataset_dir = os.path.join(config['DATASET_ROOT'], args.dataset)
dataset = read_dataset(dataset_dir).shuffle(100000).prefetch(10)

pred = 'predict'
val = 'manual'
true_positive = 0
time_residuals = []

pick_df = pd.DataFrame()

for example in dataset:
    feature = Feature(example)
    feature.filter_phase('P')
    picks = feature.picks
    pred_picks = picks.loc[picks['pick_set'] == pred]
    val_picks = picks.loc[picks['pick_set'] == val]

    for i, p in pred_picks.iterrows():
        for j, v in val_picks.iterrows():

            if validate_picks_nearby(v, p, delta=0.5):
                time_residuals.append(p['pick_time'] - v['pick_time'])

            if validate_picks_nearby(v, p, delta=0.1):
                true_positive += 1


    pick_df = pick_df.append(picks)

counts = pick_df['pick_set'].value_counts().to_dict()

precision, recall, f1 = precision_recall_f1_score(true_positive, counts[pred], counts[val])
print(f'Precision = {precision:f}, Recall = {recall:f}, F1 = {f1:f}' )

plot_error_distribution(time_residuals)
