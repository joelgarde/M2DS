author: Garde Joël
garde.joel@telecom-paristech.fr

# M2DS Optimal Transport Project



The ouput consists in 
* a pdf report discussing the paper.
* a python src folder and main script.
* additionnal data folder for convenience.

To run the code, please run main.py.
By defaults, it uses the pre-computed data.
If you want to retrain, modify train.py at set `train` to True.

Requirements are:
* numpy
* sklearn
* matplotlib
* jax
* ott-jax.

You need the reuters dataset and gloves embeddings:
* https://nlp.stanford.edu/projects/glove/
* https://kdd.ics.uci.edu/databases/reuters21578/

