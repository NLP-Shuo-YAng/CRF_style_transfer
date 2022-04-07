# Tagging Without Rewriting (TWG)
This is the codes for the paper "Tagging Without Rewriting: A Probabilistic Model to Unpaired Sentiment and Style Transfer".

## Dependencies
```
python==3.8.5
numpy==1.19.4
nltk==3.5
torch==1.8.1
torchtext==0.9.1
kenlm==0.0.0
```
Before start running the code, you should train a 5-gram LM by using kenlm and put it in the path "LM/" (Files are too big to be uploaded).

## Quick Start
Step 1: Pretrain a style classifier and a LM.

Step 2: Build a CRF and search in it.

For the above steps, run the following code:
```python
python3 train.py 
```
The outputs file will be created in the path "output.txt".

What we already have in this path is the output of the yelp dataset, for other datasets, please contact their creaters for their copyright.

The hyperparameters associated with the model structure can be found in the class Config in the file "train.py", 
but nomarlly they do not need to be adjusted manually.

The size of the parameter saved file of a trained model is approximately 248M.

## Training Time
On a single RTX 2080Ti GPU, training can usually be completed in less than an hour.
