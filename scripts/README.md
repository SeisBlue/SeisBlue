# Scripts

In the [scripts](scripts) folder:

Preprocessing:

- [stream_to_tfrecord.py](preprocessing/stream_to_tfrecord.py)
  - Make S-Files into training .tfrecord.
- [scanning.py](prototypes/scanning.py)
  - Scan through all stations available in the given time window, transfer into .pkl dataset.

Training:

- [pre_training.py](training/pre_train.py)
  - Pre-train the model using small dataset.
- [training.py](training/training.py)
  - Train the model with the pre-trained weight.
- [prediction.py](predict/predict.py)
  - Predict the probability of the picks and write into the dataset.

Evaluate:

- [plot_instance.py](visualization/plot_instance.py)
  - Plot the wavefile, picks and the probability form the .pkl dataset.
- [evalution.py](analysis/model_evaluation.py)
  - Calculate precision, recall and F1 score.
  - Plot error distribution.

Output:

- [update_picks.py](prototypes/update_picks.py)

Prototypes:

- [example_proto.py](prototypes/example_proto.py)