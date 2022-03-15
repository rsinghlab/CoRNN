# import comet_ml at the top of your file
from comet_ml import Experiment

# Create an experiment with your api key:
experiment = Experiment(
    api_key="pTKIV6STi3dM8kJmovb0xt0Ed",
    project_name="a-b-prediction",
    workspace="suchzheng2",
)
import argparse
import csv
import os
import numpy as np
import random
import re
from tqdm import tqdm
import datetime
import time
import pickle

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
# from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--cell", default=None, type=str, help="the cell line for validation and testing")
	parser.add_argument(
		"--resolution", default="100kb", type=str, help="resolution of current data")
	parser.add_argument(
		"--cross_validation", default=False, type=bool, help="use cross validation or not")
	parser.add_argument(
		"--fold", default=5, type=int, help="number of fold, usually it is 5")
	parser.add_argument(
		"--data_dir", default=None, type=str, help="data dir for loading raw data")
	parser.add_argument(
		"--add_mean", default=False, type=bool, help="add mean to data or not")

	args = parser.parse_args()

	save_path = os.path.join("data","exp_data","cla","rf")
	if not os.path.isdir(save_path):														
		os.mkdir(save_path)
	now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	run_dpath = os.path.join(save_path, "run_{}".format(now_str))
	if not os.path.isdir(run_dpath):
		os.mkdir(run_dpath)

	args.data_dir = os.path.join("data","6_cell_input_updated","6_cell_input_updated_{}".format(args.resolution))
	cells = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	# cells = ["IMR90","NHEK"]

	if args.add_mean == False:
		all_data = prepare_data_random_forest(args,cells)

	if args.cross_validation:
		all_test_result = {}
		all_runtime  = {}
		for target in cells:
			if args.add_mean:
				args.cell = target
				all_data = prepare_data_random_forest_with_mean_evec(args,cells)

			rest_cells = cells.copy().remove(target)
			
			for valid_cell in rest_cells:
				train_data = {}
				valid_data = {}
				test_data = {}

				for key,item in all_data.items():
					cell = key.split("_")[0]
					chr_num = key.split("_")[1][3:]
					if cell == target:
						test_data[key] = item
					elif cell == valid_cell:
						valid_data[key] == item
					else:
						train_data[key] = item
				
				if args.add_mean == True:
					train_x, train_y = combine_data_rf_with_mean_avg_std(train_data)
					valid_x, valid_y = combine_data_rf_with_mean_avg_std(valid_data)
					test_x, test_y = combine_data_rf_with_mean_avg_std(test_data)
				else:
					train_x, train_y = combine_data_rf_avg_std(train_data)
					valid_x, valid_y = combine_data_rf_avg_std(valid_data)
					test_x, test_y = combine_data_rf_avg_std(test_data)


		


	else:

		all_test_result = {}
		all_runtime = {}

		for target in cells:
			if args.add_mean:
				args.cell = target
				all_data = prepare_data_random_forest_with_mean_evec(args,cells)

			train_data = {}
			test_data = {}
			for key,item in all_data.items():
				cell = key.split("_")[0]
				chr_num = key.split("_")[1][3:]
				if cell == target:
					test_data[key] = item
				else:
					train_data[key] = item
			# print(train_data)

			if args.add_mean == True:
				train_x, train_y = combine_data_rf_with_mean_avg_std(train_data)
				test_x, test_y = combine_data_rf_with_mean_avg_std(test_data)
			else:
				train_x, train_y = combine_data_rf_avg_std(train_data)
				test_x, test_y = combine_data_rf_avg_std(test_data)
			# print(train_x[0])
			# print(train_y[0])
			clf = RandomForestClassifier(n_estimators = 500,random_state=0, max_features = "sqrt",verbose=3)
			start_time = time.time()
			clf.fit(train_x, train_y)
			time_cost = time.strftime("%H:%M:%S",time.gmtime(time.time()-start_time))
			print("Time cost: ",time_cost)
			
			filename = '{}_rf.sav'.format(target)
			


			file_save_path = os.path.join(run_dpath,filename)

			pickle.dump(clf, open(file_save_path, 'wb'))
			
			loaded_model = pickle.load(open(file_save_path, 'rb'))
			predicts = loaded_model.predict(test_x)
			print(predicts)
			auc = roc_auc_score(test_y, predicts)
			print("test auroc: ",auc)
			all_test_result[target] = auc
			all_runtime[target] = time_cost

		print(all_test_result)
		print(all_runtime)



if __name__ == "__main__":
    main()