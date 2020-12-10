# REBUS : Recommendation Embedding Based on freqUent Sequences
This is the original Tensorflow implementation of REBUS for the [paper](https://arxiv.org/pdf/2008.05587.pdf) and currently under review for the DAMI Journal.
Please cite our paper if you use the code or datasets.

Note that supplementary material and code use in the [paper](https://arxiv.org/pdf/2008.05587.pdf) can be found [here](https://bit.ly/39XFKe0)


## Requirements
* Python 3
* pandas
* numpy
* scipy
* tensorflow == 1.14.0
* multiprocess
* tqdm

A requirement file is available.

## Configurations
### Data
- Some Datasets can be find in the folder 01-Data
- All the datasets use in the papier can be downloaded [here](https://bit.ly/2Iyq6uf). (The most datasets come from [here](https://cseweb.ucsd.edu/~jmcauley/datasets.html) a repository created by [Julien McAuley](https://cseweb.ucsd.edu/~jmcauley/))
- Each file contains have the current format:
  > user, item, rating, time
- As the problem is Sequential Recommendation, the rating doesn't matter.
- The forlder 96-FSUB contains all frequent sequence that need REBUS. The file fsub.py in the forlder 04-Python create new one.

### Model Args (in main.py)
- model: Model to run (choices=['REBUS', 'REBUS_simple', 'REBUS_ST', 'REBUS_ST_simple', 'REBUS_LT'], required=True)
- max_iters: Max number of iterations to run (default=10000)
- quit_delta: Number of iterations at which to quit if no improvement (default=250)
- eval_freq: Frequency at which to evaluate model (default=25)
- item_per_user: Number of items test during validation (default=100)
- learning_rate: Initial learning rate (default=0.001)
- num_dims: Model dimensionality (default=10)
- mini_batch_size: Size of the mini Batch (default=128)
- max_lens: Maximun lenght for long term history (default=100)
- user_min: Number of minimal actions for a user (default=5)
- item_min: Number of minimal interaction for a item (default=5)
- min_count: Minimun times that a sequence appears in all users history (default=2)
- L: Maximun size of a sequence (default=10)
- emb_reg: L2 regularization: embbeding regularization (default=0.001)
- bias_reg: L2 regularization: Bias regularization (default=0.001)
- alpha: Alpha for control the long term (default=-1.0)
- gamma: Gamma to unified the short term and the long term. 0 equal to have only short term, 1 equal to have only long term (default=0.5)
- prediction_name_file: Prediction\'s name file (required=False)
- prediction_TopN: TopN recommandation to keep per users (default=25)
- evaluation_name_file: Evaluation\'s name file (required=False)

## Usage
```
python main.py --path path_of_dataset --model model_name
```
Full example :
```
python main.py --path 01-Data/Epinions.txt --model REBUS --max_iters 1000000 --quit_delta 250 --eval_freq 25 --item_per_user 100 --learning_rate 0.001 --num_dims 10 --mini_batch_size 128 --max_lens 105 --user_min 5 --item_min 5 --min_count 2 --L 3 --emb_reg 0.001 --bias_reg 0.001 --alpha -1.0 --gamma 0.7 --prediction_TopN 50 --prediction_name_file REBUS_Epinions_preds_preds.csv --evaluation_name_file REBUS_Epinions_preds_eval.csv
```
Important make sure that there is a frequent sequence file that match with the choosen min_count and L.
The results will be store on the following folders :
- 02-Resultats/Evaluations --> contains the evaluations
- 02-Resultats/Predictions --> contains the recommendations
