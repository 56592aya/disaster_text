# Disaster Tweets Text Analysis

**Data**

<!-- What should I expect the data format to be?

Each sample in the train and test set has the following information:

- The text of a tweet
- A keyword from that tweet (although this may be blank!)
- The location the tweet was sent from (may also be blank)

What am I predicting?

You are predicting whether a given tweet is about a real disaster or not. If so, predict a 1. If not, predict a 0.
Files

    train.csv - the training set
    test.csv - the test set
    sample_submission.csv - a sample submission file in the correct format

Columns

    id - a unique identifier for each tweet
    text - the text of the tweet
    location - the location the tweet was sent from (may be blank)
    keyword - a particular keyword from the tweet (may be blank)
    target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0) -->

## How to get started

<!--[Nothing here] -->

### Create Environmet

I used conda to create the environment in a unix environment.

```console
$ conda init
$ conda create -n ENV_NAME python=3.8.5
$ conda activate ENV_NAME
```
If you have to install torch follow the instructions [here](https://pytorch.org/get-started/locally/#linux-pip).
### Install requirements

Requirements are already put in a separate `requirements.txt` file. To install them run the following:

```console
$ pip install -r requirements.txt
```

## Running

<!--Instructions to run the code and create results-->

## Project Content

[Any specific changes to the project goes here]

```console
.
├── fig
├── input
├── notebooks
├── py.py
├── src
│   ├── config.py
│   └── utils.py
└── test
├── requirements.txt
├── README.md

```

<!-- `./app/`

- contains the information about deploying the proj as app

`./input/`

- contains the input files (raw and/or modified) used in the project

`./models/`

- contains the saved models

`./notebooks/`

- contains the notebooks for exploration and/or presentation purposes

`./src/`

- contains all the *.py codes used in the project

- `config.py`

  - consists of the environemtnal variables or global parameters used

- `create_folds.py`

  - creates the folds according to the data distribution for evaluation

- `preprocessing.py`

  - entails most of the preprocessing that goes into the raw data

- `train.py`

  - contains the tools to train the model

- `utils.py`

  - These are utility functions needed

`./requirements.txt`

- contains the libraries needed to be installed for running the programs smoothly

`./README.md`

- contains the instructions for running the programs smoothly -->

<!-- ### app
To run the Flask app in the local browser, go to the app folder.

```shell
$ export FLASK_APP=app.y
$ python app.y
```

Then open your browser at `localhost/5000` -->


### References:
