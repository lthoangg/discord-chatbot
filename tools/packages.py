import torch
import os
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
import numpy as np
import random
import json
pl.seed_everything(42, workers=True)