# Tagging without Rewriting
This is the Data set used for the paper "Tagging Without Rewriting: A Probabilistic Model to Unpaired Sentiment and Style Transfer".
## Datasets
The samples in submitted files are independent reviews. One sentence per line.

All numbers in the Yelp dataset are replaced with the '__num__' .

To obtain the GYAFC Dataset, please contact the creator for access rights:

https://github.com/raosudha89/GYAFC-corpus

For all datasets, we filtered sentences with hapax legomenons and the maximum generation length is set to 20.
## Language Models
We pretrain language models by using KenLM. 
However, they are too large to be uploaded.

Please train a language model by yourself by follow the steps at:

https://github.com/kpu/kenlm

## Language Models
We provide outputs of our model and the two most important baselines.

They can be found in the folder 'output/baselines'. Our manual evaluation is based on these files.
