# self-training-image-rec

## Setup:

Check python version 3.7.6

```
pip3 install virtualenv
virtualenv venv --python=python3
source venv/bin/activate
pip3 install -r requirements.txt
```

## Running

Running tutorial.py runs through a code tutorial of tensorflow for an image classifier for dogs and cats.
There are several different steps for dataset  viewing, training, displaying results, and finally generating some additional training data that could be used when training.
At times you have to close matplot display for the code to progress.
The code is a bit piece-wise but should serve as a good starting point for us to work from.


Running data/stl10_input.py will download the stl10 dataset and save it to disk.
It also contains various utils for loading the dataset in python.


Running Train.py will train a model using the train/test split for the stl10 dataset.


## TODO
Save / load model to disk.
Incorporate data augmentation steps featured in tutorial.py to increase performance.
Implement self-training data augmentation.