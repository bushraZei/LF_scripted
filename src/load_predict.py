import warnings
warnings.filterwarnings("ignore")


import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from read_data import read_data
from neural_nets.functions import window_dataset
from process import min_max_scale
import args
import pandas as pd


print(tf.__version__)


def prrdict(model, data, target_column):
       return model.predict(data)