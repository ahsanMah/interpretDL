#### Base Module with most necessary imports ###


######## Imports ########
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import umap
import innvestigate
import innvestigate.utils as iutils
import sys

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# Converting labels to 1-Hot Vectors
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##########################

root_logdir = "./tf_logs"
datadir = "data/"
figures_dir = "data/figures/"

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 16

# To make notebooks' output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
np.random.seed(seed=42) 

print(tf.__version__)


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
