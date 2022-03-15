# import comet_ml at the top of your file
# from comet_ml import Experiment

# # Create an experiment with your api key:
# experiment = Experiment(
#     api_key="pTKIV6STi3dM8kJmovb0xt0Ed",
#     project_name="a-b-prediction",
#     workspace="suchzheng2",
# )
import argparse
import csv
import os
import sys
import numpy as np
import random
import re
from tqdm import tqdm
import datetime
import time
import pickle
import itertools

from preprocess import *
from model import *
from utils import *
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
# from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import logging

save_path = os.path.join("data","exp_data","cla","rf")
if not os.path.isdir(save_path):														
	os.mkdir(save_path)
now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
run_dpath = os.path.join(save_path, "run_{}".format(now_str))
if not os.path.isdir(run_dpath):
	os.mkdir(run_dpath)

a_logger = logging.getLogger()
a_logger.setLevel(logging.DEBUG)

output_file_handler = logging.FileHandler(os.path.join(run_dpath,"output.log"))
stdout_handler = logging.StreamHandler(sys.stdout)

a_logger.addHandler(output_file_handler)
a_logger.addHandler(stdout_handler)



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--cell", default=None, type=str, help="the cell line for validation and testing")
	parser.add_argument(
		"--resolution", default="100kb", type=str, help="resolution of current data")
	parser.add_argument(
		"--cross_validation", default=True, type=bool, help="use cross validation or not")
	parser.add_argument(
		"--fold", default=5, type=int, help="number of fold, usually it is 5")
	parser.add_argument(
		"--data_dir", default=None, type=str, help="data dir for loading raw data")
	parser.add_argument(
		"--add_mean", default=True, type=bool, help="add mean to data or not")
	parser.add_argument(
		"--add_std", default=True, type=bool, help="add std to data or not")
	parser.add_argument(
		"--use_sgd", default=False, type=bool, help="use sgd model or not")

	args = parser.parse_args()

	args.data_dir = os.path.join("data","6_cell_input_updated","6_cell_input_updated_{}".format(args.resolution))


	targets = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	cells = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]


	if not args.add_std:
		if not args.add_mean:
			model_dir = os.path.join("data","exp_data","cla","rf","??")
		if args.add_mean:
			model_dir = os.path.join("data","exp_data","cla","rf","??")
	if args.add_std:
		if not args.add_mean:
			model_dir = os.path.join("data","exp_data","cla","rf","run_2021-08-14-15-33-13")
		if args.add_mean:
			model_dir = os.path.join("data","exp_data","cla","rf","run_2021-08-13-23-31-35")	


	if args.add_mean == False:
		all_data = prepare_data_random_forest(args,cells)

	test_results = []
 
	for target in targets:

		args.cell = target

		if args.add_mean == True:
			all_data = prepare_data_random_forest_with_mean_evec(args,cells)
		

		test_data = dict.fromkeys(cells)
		for key in test_data.keys():
			test_data[key] = {}

		for key, data in all_data.items():
			cell = key.split("_")[0]
			test_data[cell][key] = data
	 
		
		target_model_path = os.path.join(model_dir,"{}_lr.sav".format(target))

		model = pickle.load(open(target_model_path, 'rb'))
		model_test_data = test_data[target]

		if args.add_std:
			if args.add_mean == True:
				test_x, test_y = combine_data_with_mean_avg_std(model_test_data)
			else:
				test_x, test_y = combine_data_avg_std(model_test_data)
		if not args.add_std:
			if args.add_mean == True:
				test_x, test_y = combine_data_with_mean_avg(model_test_data)
			else:
				test_x, test_y = combine_data_avg(model_test_data)


		test_predict = model.predict(test_x)
		test_auc = roc_auc_score(test_y, test_predict)

		test_results.append(test_auc)
		
		a_logger.debug("logistic regression of {}: {}~~~~~~".format(target,test_auc))

		
	print_line = ''
	a_logger.debug("Testing results:")
	for auc in test_results:
		print_line += "{:.3f},".format(auc)
	print(targets)
	print("auc: {}".format(print_line))





if __name__ == "__main__":
    main()