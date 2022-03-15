

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold

from collections import defaultdict
import os
import csv
import datetime
import json



def AUROC():
	targets = [1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1]
	# predict = [0.9,0.8,0.7,0.2,0.2,1.0,0.9,0.8,0.3,0.3,0.4,0.9,0.1,0.1,0.9,0.7,0.9,0.3,0.9]
	predict = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
	# print("targets: ",targets)
	# print("predict: ",predict)
	auc = roc_auc_score(targets, predict)
	
	fpr, tpr, _ = roc_curve(targets, predict)
	# experiment.log_metric("AUROC",auc)
	# experiment.log_curve(curve_name,fpr,tpr)

	print("auc",auc)
	print("fpr",fpr)
	print("tpr",tpr)
	pyplot.plot(fpr, tpr, linestyle='--', label= "{:.3f}".format(auc))
	# axis labels
	pyplot.title("AUROC")
	pyplot.xlabel('False Positive Rate')
	pyplot.ylabel('True Positive Rate')
	# show the legend
	pyplot.legend()
	# pyplot.savefig(img_path)
	pyplot.clf()
	pyplot.show()

AUROC()