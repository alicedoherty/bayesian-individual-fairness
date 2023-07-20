# Individual Fairness in Bayesian Neural Networks

The following repository contains the code related to my Final Year Project exploring individual fairness in Bayesian neural networks (BNNs).

## Contents

- `IndividualFairnessinBNNs_AliceDoherty.pdf` - Final Year Project/Dissertation report.
- `run_trial.py` - Main script for running an experimental trial (i.e. training a variety of BNNs and deterministic NNs on the Adult dataset and evaluating their individual fairness).
- `visualise_data.py` - Script for visualising the results of the experiments.
- `requirements.txt` - Requirements required to run the provided code. Use `pip install -r requirements.txt`.
- `BNN_FGSM.py` and `DNN_FGSM.py` - Custom _Fair-FGSM_ implementation developed to measure individual fairness in BNNs and deterministic NNs respectively.
- `\deepbayes` - Directory containing the minorly modified Deepbayes Python library. The original Deepbayes source code can be found [here](https://github.com/matthewwicker/deepbayes).
- `\data` - Directory containing the Adult dataset. The original dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/adult).
- `\results` - Directory containing the results of the experimental trials.
- `\heatmaps`, `\fairness_acc_plots`, `\epsilon_plots` - Directories containing the graphs visualising the experimental results.
