Scripts
=======

In the `scripts`_ folder:

Preprocessing:

-  `stream_to_tfrecord.py`_

   -  Make S-Files into training .tfrecord.

-  `scanning.py`_

   -  Scan through all stations available in the given time window,
      transfer into .pkl dataset.

Training:

-  `pre_training.py`_

   -  Pre-train the model using small dataset.

-  `training.py`_

   -  Train the model with the pre-trained weight.

-  `prediction.py`_

   -  Predict the probability of the picks and write into the dataset.

Evaluate:

-  `plot_instance.py`_

   -  Plot the wavefile, picks and the probability form the .pkl
      dataset.

-  `evalution.py`_

   -  Calculate precision, recall and F1 score.
   -  Plot error distribution.

Output:

-  `update_picks.py`_

Prototypes:

-  `example_proto.py`_

.. _scripts: scripts
.. _stream_to_tfrecord.py: preprocessing/stream_to_tfrecord.py
.. _scanning.py: prototypes/scanning.py
.. _pre_training.py: training/pre_train.py
.. _training.py: training/training.py
.. _prediction.py: predict/predict.py
.. _plot_instance.py: visualization/plot_instance.py
.. _evalution.py: analysis/model_evaluation.py
.. _update_picks.py: prototypes/update_picks.py
.. _example_proto.py: prototypes/example_proto.py