import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision
import torch.utils.data


import datasets

import hparams_registry
import algorithms
from lib import misc
from lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__" :
    # hparams = hparams_registry.default_hparams("ERM", "PACS")        
    # dataset = vars(datasets)["PACS"]("/home1/durga/sainath/Robust Distillation/KD/DomainBed/domainbed/data", [1], hparams)

    # print(f"Number of environments : {len(dataset.datasets)}")
    # print(f"Number of classes : {dataset.num_classes}")