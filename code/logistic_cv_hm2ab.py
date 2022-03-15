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



def SGD_cross_validation(all_data, select_param, cells, args):
	a_logger.debug("==================================================== \nTarget: {} \nbootstrap = {}\nn_estimators = {}\nmax_depth = {}\nmax_features = {} \nmin_samples_split = {}\nmin_samples_leaf = {}\n"
			.format(args.cell,select_param['bootstrap'],select_param['n_estimators'],\
				select_param['max_depth'],select_param['max_features'],\
				select_param['min_samples_split'],select_param['min_samples_leaf']))
	a_logger.debug("====================================================")

	other_cells = cells.copy()
	other_cells.remove(args.cell)

	cross_validation_results = {}
	best_fold = 0
	best_valid_auc = 0
	best_model_test_auc = 0

	for fold_id, valid_cell in enumerate(other_cells):
		a_logger.debug("\n----- fold {}, test cell:{}, valid cell:{} -----\n".format(fold_id,args.cell,valid_cell))
		cross_validation_results[fold_id] = {}
		cross_validation_results[fold_id]["valid_cell"] = valid_cell
		train_data = {}
		valid_data = {}
		test_data = {}

		for key,item in all_data.items():
			cell = key.split("_")[0]
			chr_num = key.split("_")[1][3:]
			if cell == args.cell:
				test_data[key] = item
			elif cell == valid_cell:
				valid_data[key] = item
			else:
				train_data[key] = item
		if args.use_600 == True:
			train_x, train_y = combine_data_rf(train_data)
			valid_x, valid_y = combine_data_rf(valid_data)
			test_x, test_y = combine_data_rf(test_data)
		elif args.add_std == True:
			if args.add_mean == True:
				train_x, train_y = combine_data_with_mean_avg_std(train_data)
				valid_x, valid_y = combine_data_with_mean_avg_std(valid_data)
				test_x, test_y = combine_data_with_mean_avg_std(test_data)
			else:
				train_x, train_y = combine_data_avg_std(train_data)
				valid_x, valid_y = combine_data_avg_std(valid_data)
				test_x, test_y = combine_data_avg_std(test_data)
		else:
			if args.add_mean == True:
				train_x, train_y = combine_data_with_mean_avg(train_data)
				valid_x, valid_y = combine_data_with_mean_avg(valid_data)
				test_x, test_y = combine_data_with_mean_avg(test_data)
			else:
				train_x, train_y = combine_data_avg(train_data)
				valid_x, valid_y = combine_data_avg(valid_data)
				test_x, test_y = combine_data_avg(test_data)

		model = new_lr_model(select_param,args)

		start_time = time.time()

		model.fit(train_x, train_y)
		time_cost = time.strftime("%H:%M:%S",time.gmtime(time.time()-start_time))
		a_logger.debug("Time cost: {}".format(time_cost))

		valid_predict = model.predict(valid_x)
		valid_auc = roc_auc_score(valid_y, valid_predict)
		a_logger.debug("Validation Result: {}".format(valid_auc))

		cross_validation_results[fold_id]["model"] = model
		cross_validation_results[fold_id]["valid_result"] = valid_auc

		if valid_auc > best_valid_auc:
			best_valid_auc = valid_auc
			best_fold = fold_id

			test_predict = model.predict(test_x)
			best_model_test_auc = roc_auc_score(test_y, test_predict)

	a_logger.debug("===best validation auc: {}===".format(best_valid_auc))

	return cross_validation_results[best_fold]["model"], best_valid_auc, best_model_test_auc

def logistic_regression_cross_validation(all_data, select_dict, cells, args):
	a_logger.debug("linear regresion corss validation...")

	other_cells = cells.copy()
	other_cells.remove(args.cell)

	cross_validation_results = {}
	best_fold = 0
	best_valid_auc = 0
	best_model_test_auc = 0

	for fold_id, valid_cell in enumerate(other_cells):
		a_logger.debug("\n----- fold {}, test cell:{}, valid cell:{} -----\n".format(fold_id,args.cell,valid_cell))
		cross_validation_results[fold_id] = {}
		cross_validation_results[fold_id]["valid_cell"] = valid_cell
		train_data = {}
		valid_data = {}
		test_data = {}

		for key,item in all_data.items():
			cell = key.split("_")[0]
			chr_num = key.split("_")[1][3:]
			if cell == args.cell:
				test_data[key] = item
			elif cell == valid_cell:
				valid_data[key] = item
			else:
				train_data[key] = item


		if args.use_600 == True:
			train_x, train_y = combine_data_rf(train_data)
			valid_x, valid_y = combine_data_rf(valid_data)
			test_x, test_y = combine_data_rf(test_data)
		elif args.add_std == True:
			if args.add_mean == True:
				train_x, train_y = combine_data_with_mean_avg_std(train_data)
				valid_x, valid_y = combine_data_with_mean_avg_std(valid_data)
				test_x, test_y = combine_data_with_mean_avg_std(test_data)
			else:
				train_x, train_y = combine_data_avg_std(train_data)
				valid_x, valid_y = combine_data_avg_std(valid_data)
				test_x, test_y = combine_data_avg_std(test_data)
		else:
			if args.add_mean == True:
				train_x, train_y = combine_data_with_mean_avg(train_data)
				valid_x, valid_y = combine_data_with_mean_avg(valid_data)
				test_x, test_y = combine_data_with_mean_avg(test_data)
			else:
				train_x, train_y = combine_data_avg(train_data)
				valid_x, valid_y = combine_data_avg(valid_data)
				test_x, test_y = combine_data_avg(test_data)

		model = new_lr_model(select_dict,args)

		start_time = time.time()

		model.fit(train_x, train_y)
		time_cost = time.strftime("%H:%M:%S",time.gmtime(time.time()-start_time))
		a_logger.debug("Time cost: {}".format(time_cost))

		valid_predict = model.predict(valid_x)
		valid_auc = roc_auc_score(valid_y, valid_predict)
		a_logger.debug("Validation Result: {}".format(valid_auc))

		cross_validation_results[fold_id]["model"] = model
		cross_validation_results[fold_id]["valid_result"] = valid_auc

		if valid_auc > best_valid_auc:
			best_valid_auc = valid_auc
			best_fold = fold_id

			test_predict = model.predict(test_x)
			best_model_test_auc = roc_auc_score(test_y, test_predict)

		

	a_logger.debug("===best validation auc: {}===".format(best_valid_auc))

	return cross_validation_results[best_fold]["model"], best_valid_auc, best_model_test_auc


def new_rf_model(select_dict):
	clf = RandomForestClassifier(
				bootstrap = select_dict['bootstrap'],\
				n_estimators = select_dict['n_estimators'],\
				max_depth = select_dict['max_depth'],\
				max_features = select_dict['max_features'],\
				min_samples_split = select_dict['min_samples_split'],\
				min_samples_leaf = select_dict['min_samples_leaf'],
				random_state=0,\
				verbose=1)

	return clf


def new_lr_model(select_dict, args):
	print("creating new LogisticRegression model")
		
	clf = LogisticRegression(
		penalty = select_dict['penalty'],\
		solver = select_dict['solver'],\
		C = select_dict['C'],\
		class_weight = select_dict['class_weight'],\
		max_iter = select_dict['max_iter'],\
		random_state=0,\
		verbose=1)

	return clf



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
		"--add_mean", default=False, type=bool, help="add mean to data or not")
	parser.add_argument(
		"--add_std", default=False, type=bool, help="add std to data or not")
	parser.add_argument(
		"--search", default=True, type=bool, help="search hyperparameter or not")
	parser.add_argument(
		"--use_600", default=True, type=bool, help="use 600 features")


	args = parser.parse_args()

	args.data_dir = os.path.join("data","6_cell_input_updated","6_cell_input_updated_{}".format(args.resolution))


	targets = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	cells = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]

	hyperparameter = {
		'penalty': ['l1','l2'],
		'C':[0.0001, 1, 100, 1000],
		'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
		'class_weight':['balanced', None],
		'max_iter':[1, 10, 100, 500],
	}


	print("Here!!")


	if args.add_mean == False:
		all_data = prepare_data_random_forest(args,cells)

	grid_search_results = {}
 
	for target in targets:
		args.cell = target
		best_auc = 0
		best_param = {}
		best_model_test_auc = 0
		grid_search_results[target] = {}

		if args.add_mean == True:
			all_data = prepare_data_random_forest_with_mean_evec(args,cells)


		total_search = len(list(itertools.product(*hyperparameter.values())))

		if args.search:
			for idx, param_selection in enumerate(list(itertools.product(*hyperparameter.values()))):
				a_logger.debug("Search Progress: {}/{}".format(idx,total_search))

				select_dict = dict(zip(hyperparameter.keys(),param_selection))
				if select_dict["penalty"] == "l1" and \
					select_dict['solver'] != "saga" and \
					select_dict['solver'] != "liblinear":
					a_logger.debug("skip setting: {}".format(select_dict))
					continue

				clf, valid_auc, model_test_auc= logistic_regression_cross_validation(all_data,select_dict,cells,args)
				if valid_auc > best_auc:
					best_auc = valid_auc
					best_param = select_dict

					filename = '{}.sav'.format(target)
					file_save_path = os.path.join(run_dpath,filename)
					pickle.dump(clf, open(file_save_path, 'wb'))

					param_file = '{}_model_config.json'.format(target)
					param_save_path = os.path.join(run_dpath, param_file)
					with open(param_save_path, 'w') as outfile:
						json.dump(select_dict, outfile)
					best_model_test_auc = model_test_auc

			grid_search_results[target]["param"] = best_param
			grid_search_results[target]["valid_auc"] = best_auc
			grid_search_results[target]["test_auc"] = best_model_test_auc

			print("~~~~~Logistic Regression Grid search of {}: {}~~~~~~".format(target,grid_search_results[target]["test_auc"]))

		else:
			clf, valid_auc, model_test_auc= logistic_regression_cross_validation(all_data,cells,args)
			filename = '{}.sav'.format(target)
			file_save_path = os.path.join(run_dpath,filename)
			pickle.dump(clf, open(file_save_path, 'wb'))

			# param_file = '{}_model_config.json'.formatS(target)
			# param_save_path = os.path.join(run_dpath, param_file)
			# with open(param_save_path, 'w') as outfile:
			# 	json.dump(select_dict, outfile)

			# grid_search_results[target]["param"] = best_param
			grid_search_results[target]["valid_auc"] = valid_auc
			grid_search_results[target]["test_auc"] = model_test_auc

			print("~~~~~linear regression of {}: {}~~~~~~".format(target,grid_search_results[target]["test_auc"]))

	

	a_logger.debug("******Grid search and cross_validation result******")
	for key, results in grid_search_results.items():
		a_logger.debug("cell: {}, valid: {}, test: {}".format(key,results["valid_auc"],results["test_auc"]))
		# a_logger.debug("param: {}".format(results["param"]))


if __name__ == "__main__":
    main()