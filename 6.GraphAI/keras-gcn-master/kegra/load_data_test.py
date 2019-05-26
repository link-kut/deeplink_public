from __future__ import print_function

from kegra.utils import *

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Define parameters
DATASET = 'cora'

# Get data
X, A, labels = load_data(dataset=DATASET)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(labels)

# Normalize X
X /= X.sum(axis=1).reshape(-1, 1)