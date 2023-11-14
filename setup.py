import torch
import numpy as np
from sklearn.datasets import make_classification, make_gaussian_quantiles
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from dgllife.data import Tox21
from dgllife.utils import SMILESToBigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm