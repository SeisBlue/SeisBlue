In the [scripts](scripts) folder:

Preprocessing:
 
- [training_set_to_tfrecord.py](preprocessing/training_set_to_tfrecord.py)
  - Make S-Files into training .tfrecord.
- [scanning.py](preprocessing/scanning.py)
  - Scan through all stations available in the given time window, transfer into .pkl dataset.
  
Training:

- [pre_training.py](training/pre_train.py)
  - Pre-train the model using small dataset.
- [training.py](training/training.py)
  - Train the model with the pre-trained weight.
- [prediction.py](predict/predict.py)
  - Predict the probability of the picks and write into the dataset.

Evaluate:

- [plot_instance.py](evaluation/plot_instance.py)
  - Plot the wavefile, picks and the probability form the .pkl dataset.
- [evalution.py](evaluation/evalution.py)
  - Calculate precision, recall and F1 score.
  - Plot error distribution.

Output:

- [update_picks.py](output/update_picks.py)

Prototypes:

- [example_proto.py](prototypes/example_proto.py)