from comet_ml import Experiment

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
from sklearn.metrics import roc_auc_score
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

from preprocess import *
#from correlation import calculate_correlation


def AUROC(experiment, targets, predict, img_path, curve_name,save_fig):
	# targets = [1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1]
	# predict = [0.9,0.8,0.7,0.2,0.2,1.0,0.9,0.8,0.3,0.3,0.4,0.9,0.1,0.1,0.9,0.7,0.9,0.3,0.9]
	# print("targets: ",targets)
	# print("predict: ",predict)
	auc = roc_auc_score(targets, predict)
	
	fpr, tpr, _ = roc_curve(targets, predict)
	# experiment.log_metric("AUROC",auc)
	experiment.log_curve(curve_name,fpr,tpr)

	if save_fig:
		# plot the roc curve for the model
		pyplot.plot(fpr, tpr, linestyle='--', label= "{}: {:.3f}".format(curve_name,auc))
		# axis labels
		pyplot.title("AUROC")
		pyplot.xlabel('False Positive Rate')
		pyplot.ylabel('True Positive Rate')
		# show the legend
		pyplot.legend()
		pyplot.savefig(img_path)
		pyplot.clf()

		experiment.log_image(img_path)
	return auc,fpr,tpr



		#experiment.log_image(img_path)

def AUPRC(experiment, targets, predict, img_path, curve_name,save_fig):
	precision, recall, _ = precision_recall_curve(targets, predict)
	auc_val = auc(recall, precision)
	# experiment.log_metric("AUPRC",auc_val)
	experiment.log_curve(curve_name,recall, precision)


	if save_fig:
		pyplot.plot(recall, precision, marker='.', label= "{}: {:.3f}".format(curve_name,auc_val))
		# axis labels
		pyplot.title("AUPRC")
		pyplot.xlabel('Recall')
		pyplot.ylabel('Precision')
		# show the legend
		pyplot.legend()
		pyplot.savefig(img_path)
		pyplot.clf()

		experiment.log_image(img_path)
	return auc_val,recall,precision
		# experiment.log_image(img_path)


def F1(experiment, targets, predict):
	f_1 = f1_score(targets,predict)
	experiment.log_metric("F1",f_1)

def toBinary(val):
	if float(val) > 0:
		return str(1)
	else:
		return str(0)

def evaluate_regression(experiment, targets, predicts):

	mean_abs_err = mean_absolute_error(targets, predicts)
	experiment.log_metric("mean_abs_err",mean_abs_err)

	mean_sqr_err = mean_squared_error(targets, predicts)
	experiment.log_metric("mean_sqr_err",mean_sqr_err)

	# mean_sqr_log_err = mean_squared_log_error(targets, predicts)
	# experiment.log_metric("mean_sqr_log_err",mean_sqr_log_err)

	med_abs_err = median_absolute_error(targets, predicts)
	experiment.log_metric("med_abs_err",med_abs_err)

	r2 = r2_score(targets, predicts)
	experiment.log_metric("r2",r2)

	exp_var_score = explained_variance_score(targets, predicts)
	experiment.log_metric("exp_var_score",exp_var_score)
	


def prepare_data_cell_lines_v6(args, data_dir,target_cell, use_corre = False):
	'''
	Version 6

	Use chr 1-16 of all cell lines for training
	Use chr 17-18 of IMR90 for validation

	use mean value to fill up missing values

	'''
	cell_lines = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	# cell_lines = ["NHEK"]
	
	# read all data, store with key cellName_chrX: all data
	all_data = {}
	train_data_all = {}
	test_data_all = {}

	correlations = {}

	corre_dpath = os.path.join("data","6_cell_{}_correlations.json".format(args.resolution))

	if args.use_corre:
		if os.path.isfile(corre_dpath):
			with open(corre_dpath, 'r') as j:
				correlations = json.loads(j.read())
		else:
			for cell in cell_lines:
				correlations[cell] = calculate_correlation(cell)
			with open(corre_dpath, 'w') as outfile:
				json.dump(correlations, outfile)

	for cell_line in cell_lines:
		cell_line_dir = os.path.join(args.data_dir,cell_line)
		for filename in os.listdir(cell_line_dir):
			print("***Reading data from {}...".format(filename))
			chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as csvfile:
				data_reader = csv.reader(csvfile, delimiter=',')
				lines = []
				for row in data_reader:
					row = [val if val else 'NA' for val in row]
					lines.append(row)
				all_data[chromosome] = lines

	# compute mean value for each histone modification, replace NA with mean

	data_info = {0:{"sum":0, "count":0},\
				1:{"sum":0, "count":0},\
				2:{"sum":0, "count":0},\
				3:{"sum":0, "count":0},\
				4:{"sum":0, "count":0},\
				5:{"sum":0, "count":0}
				}
	empty_point = defaultdict(list)

	for chrom, data in all_data.items():
		for line_id, line in enumerate(data):
			for hm_idx, value in enumerate(line[3:]):
				# print(hm_idx)
				# print(value)
				if value == 'NA':
					empty_point[hm_idx].append((chrom,line_id))
				else:
					sum_value = data_info[hm_idx]["sum"] + float(value)
					data_count = data_info[hm_idx]["count"] + 1
					data_info[hm_idx]["sum"] = sum_value
					data_info[hm_idx]["count"] = data_count

	mean_values = {}
	print(data_info)
	for hm_idx, info in data_info.items():
		# print(hm_idx,info)
		mean_values[hm_idx] = info["sum"] / info["count"]

	# print(mean_values)

	for hm_idx, point_list in empty_point.items():
		mean_value = mean_values[hm_idx]
		for (chrom, line_id) in point_list:
			all_data[chrom][line_id][hm_idx+3] = mean_value

	print("***Finish Imputing***")

	#clean data
	cleaned_data = {}
	for key, item in all_data.items():
		cell = key.split("_")[0]
		chr_num = key.split("_")[1]
		if use_corre:
			corre = correlations[cell][chr_num]
		else:
			corre = None
		cleaned_data[key] = clean_data(item,corre)

	#divide data into train, validate, testing	
	train_chrom = {}
	validate_chrom = {}
	test_chrom = {}

	print("***Finish Cleaning***")

	for chrom, data in cleaned_data.items():
		# print(chrom)
		cell = chrom.split("_")[0]
		chr_num = chrom.split("_")[1][3:]
		# print("target cell: {}".format(target_cell))
		# print(cell)
		# print(chr_num)

		if chr_num == "X":
			continue
		chr_num	 = int(chr_num)	
		if cell == args.cell:
			if chr_num >= 17 and chr_num < 19:
				# print("add valid data")
				validate_chrom[chrom] = data
			if chr_num >= 19 and chr_num < 23:
				# print("add test data")
				test_chrom[chrom] = data
		else:
			if chr_num < 17:
				# print("add train data")
				train_chrom[chrom] = data
	print("***Finish Combining***")
	# print(test_chrom.keys())
	#combine data
	training_data = combine_data(train_chrom)
	validation_data = combine_data(validate_chrom)
	testing_data = combine_data(test_chrom)

	print(training_data[0])


	return training_data, validation_data, testing_data


def prepare_data_cell_lines_v7(args, data_dir,target_cell, use_corre = False):
	'''
	Version 6

	Use chr 1-16 of all cell lines for training
	Use chr 17-18 of IMR90 for validation

	use mean value to fill up missing values

	'''
	cell_lines = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	# cell_lines = ["NHEK"]
	
	# read all data, store with key cellName_chrX: all data
	all_data = {}
	train_data_all = {}
	test_data_all = {}

	correlations = {}

	corre_dpath = os.path.join("data","6_cell_{}_correlations.json".format(args.resolution))

	if args.use_corre:
		if os.path.isfile(corre_dpath):
			with open(corre_dpath, 'r') as j:
				correlations = json.loads(j.read())
		else:
			for cell in cell_lines:
				correlations[cell] = calculate_correlation(cell)
			with open(corre_dpath, 'w') as outfile:
				json.dump(correlations, outfile)

	for cell_line in cell_lines:
		cell_line_dir = os.path.join(args.data_dir,cell_line)
		for filename in os.listdir(cell_line_dir):
			print("***Reading data from {}...".format(filename))
			chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as csvfile:
				data_reader = csv.reader(csvfile, delimiter=',')
				lines = []
				for row in data_reader:
					row = [val if val else 'NA' for val in row]
					lines.append(row)
				all_data[chromosome] = lines

	# compute mean value for each histone modification, replace NA with mean

	data_info = {0:{"sum":0, "count":0},\
				1:{"sum":0, "count":0},\
				2:{"sum":0, "count":0},\
				3:{"sum":0, "count":0},\
				4:{"sum":0, "count":0},\
				5:{"sum":0, "count":0}
				}
	empty_point = defaultdict(list)

	for chrom, data in all_data.items():
		for line_id, line in enumerate(data):
			for hm_idx, value in enumerate(line[3:]):
				# print(hm_idx)
				# print(value)
				if value == 'NA':
					empty_point[hm_idx].append((chrom,line_id))
				else:
					sum_value = data_info[hm_idx]["sum"] + float(value)
					data_count = data_info[hm_idx]["count"] + 1
					data_info[hm_idx]["sum"] = sum_value
					data_info[hm_idx]["count"] = data_count

	mean_values = {}
	print(data_info)
	for hm_idx, info in data_info.items():
		# print(hm_idx,info)
		mean_values[hm_idx] = info["sum"] / info["count"]

	# print(mean_values)

	for hm_idx, point_list in empty_point.items():
		mean_value = mean_values[hm_idx]
		for (chrom, line_id) in point_list:
			all_data[chrom][line_id][hm_idx+3] = mean_value

	print("***Finish Imputing***")

	#clean data
	cleaned_data = {}
	for key, item in all_data.items():
		cell = key.split("_")[0]
		chr_num = key.split("_")[1]
		if use_corre:
			corre = correlations[cell][chr_num]
		else:
			corre = None
		cleaned_data[key] = clean_data(item,corre)

	#divide data into train, validate, testing	
	train_chrom = {}
	validate_chrom = {}
	test_chrom = {}

	print("***Finish Cleaning***")

	for chrom, data in cleaned_data.items():
		# print(chrom)
		cell = chrom.split("_")[0]
		chr_num = chrom.split("_")[1][3:]
		# print("target cell: {}".format(target_cell))
		# print(cell)
		# print(chr_num)

		if chr_num == "X":
			continue
		chr_num	 = int(chr_num)	
		if cell == args.cell:
			if chr_num >= 17 and chr_num < 19:
				# print("add valid data")
				validate_chrom[chrom] = data
			else:
				# print("add test data")
				test_chrom[chrom] = data
		else:
			train_chrom[chrom] = data
	print("***Finish Combining***")
	# print(test_chrom.keys())
	#combine data
	training_data = combine_data(train_chrom)
	validation_data = combine_data(validate_chrom)
	testing_data = combine_data(test_chrom)

	print(training_data[0])


	return training_data, validation_data, testing_data


def prepare_test_data_by_region(args,cell_lines):

	all_data = {}
	train_data_all = {}
	test_data_all = {}

	for cell_line in cell_lines:
		cell_line_dir = os.path.join(args.data_dir,cell_line)
		for filename in os.listdir(cell_line_dir):
			print("***Reading data from {}...".format(filename))
			chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as csvfile:
				data_reader = csv.reader(csvfile, delimiter=',')
				lines = []
				for row in data_reader:
					row = [val if val else 'NA' for val in row]
					lines.append(row)
				all_data[chromosome] = lines

	# compute mean value for each histone modification, replace NA with mean
	data_info = {0:{"sum":0, "count":0},\
				1:{"sum":0, "count":0},\
				2:{"sum":0, "count":0},\
				3:{"sum":0, "count":0},\
				4:{"sum":0, "count":0},\
				5:{"sum":0, "count":0}
				}
	empty_point = defaultdict(list)

	for chrom, data in all_data.items():
		for line_id, line in enumerate(data):
			for hm_idx, value in enumerate(line[3:]):
				# print(hm_idx)
				# print(value)
				if value == 'NA':
					empty_point[hm_idx].append((chrom,line_id))
				else:
					sum_value = data_info[hm_idx]["sum"] + float(value)
					data_count = data_info[hm_idx]["count"] + 1
					data_info[hm_idx]["sum"] = sum_value
					data_info[hm_idx]["count"] = data_count

	mean_values = {}
	print(data_info)
	for hm_idx, info in data_info.items():
		# print(hm_idx,info)
		mean_values[hm_idx] = info["sum"] / info["count"]

	# print(mean_values)

	for hm_idx, point_list in empty_point.items():
		mean_value = mean_values[hm_idx]
		for (chrom, line_id) in point_list:
			all_data[chrom][line_id][hm_idx+3] = mean_value

	print("***Finish Imputing***")

	#load mean eigenvector
	# mean_evec = {}
	# mean_evec_dir = os.path.join("data","mean_evec","mean_evec_updated_{}_allchromosome_by_chromosome".format(args.resolution))
	#load mean eigenvector
	
	mean_evec = {}
	mean_evec_dir = os.path.join("data","mean_evec","mean_evec_updated_100kb_allchromosome_by_chromosome")

	cleaned_data = {}

	if args.add_mean_evec == True:
		for cell in cell_lines:
			for ch in range(1,23):
				load_path = os.path.join(mean_evec_dir,cell,"{}_chr{}_test.txt".format(cell,ch))
				f = open(load_path, "r")
				all_mean = f.readlines()
				all_mean = [mean.strip().lower() for mean in all_mean]
				
				#now load updated eigenvector
				updated_evec_file = os.path.join("data","updated_eigenvectors_100kb",cell,\
					"{}_eigen_100kb_chr{}.csv".format(cell,ch))
				f = open(updated_evec_file, "r")
				lines = f.readlines()
				eigenvectors = lines[1:]

				filter_means = []
				for idx, evec in enumerate(eigenvectors):
					evec = evec.split(',')[1].strip().lower()
					if evec != "":
						filter_means.append(all_mean[idx])

				mean_evec["{}_chr{}".format(cell,ch)] = filter_means
		#clean data
		# cleaned_data = {}
		for key, item in all_data.items():
			cell = key.split("_")[0]
			chr_num = key.split("_")[1]
			if args.use_corre:
				corre = correlations[cell][chr_num]
			else:
				corre = None
			curr_mean_evec = mean_evec[key]
			cleaned_data[key] = clean_data_with_mean(item,corre,curr_mean_evec, True)
	else:
		for key, item in all_data.items():
			cell = key.split("_")[0]
			chr_num = key.split("_")[1]
			cleaned_data[key] = clean_data(item,None)



	region_dir = os.path.join("data","region","region_{}_by_chromosome".format(args.resolution))
	updated_evec_dir = os.path.join("data","updated_eigenvectors_{}".format(args.resolution))
	#generate test data for mean eigenvector

	for cell in cell_lines:
		region_data = {}
		test_data = {}
		for chr_num in range(1,23):
			#load training input data
			print("{}: chr {}".format(cell,chr_num))
			nan_mean = 0
			key_name = "{}_chr{}".format(cell,chr_num)
			data = cleaned_data[key_name]

			#load region file
			region_file_path = os.path.join(region_dir,cell,"region_{}_chr{}.txt".format(cell,chr_num))
			region_f = open(region_file_path,"r")
			region_lines = region_f.readlines()
			region_lines = [float(line.strip().lower()) for line in region_lines]

			#load updated eigenvector
			updated_evec_path = os.path.join(updated_evec_dir,cell,"{}_eigen_{}_chr{}.csv".format(cell,args.resolution,chr_num))
			evec_f = open(updated_evec_path,"r")
			evec_lines = evec_f.readlines()
			evec_lines = evec_lines[1:]
			evec_lines = [line.split(',')[1].strip().lower() for line in evec_lines]

			#load mean eigenvector
			mean_evec_path = os.path.join("data","mean_evec","mean_evec_updated_100kb_allchromosome_by_chromosome",\
				cell,"{}_chr{}_test.txt".format(cell,chr_num))
			mean_evec_f = open(mean_evec_path,"r")
			mean_evec_lines = mean_evec_f.readlines()
			mean_evec_lines = [line.strip().lower() for line in mean_evec_lines]


			filtered_region = []
			filtered_mean_evec = []
			filtered_evec = []
			for idx,item in enumerate(evec_lines[:-1]):
				if item !="":	
					filtered_region.append(region_lines[idx])
					filtered_mean_evec.append(mean_evec_lines[idx])
					filtered_evec.append(item)



			print("length of mean evec:	",len(filtered_mean_evec))
			print("length of region:	",len(filtered_region))
			print("number of nan mean:	",nan_mean)

			for idx,region_val in enumerate(filtered_region):
				if region_val not in region_data.keys():
					region_data[region_val] = {}
					region_data[region_val]["mean_evec"] = []
					region_data[region_val]["evec"] = []
					region_data[region_val]["test_data"] = []

				if filtered_mean_evec[idx] != "nan":
					region_data[region_val]["mean_evec"].append(filtered_mean_evec[idx])
					region_data[region_val]["evec"].append(str(1) if float(filtered_evec[idx])>0 else str(0))
					region_data[region_val]["test_data"].append(data[idx])
				# if region_val == 1: #static
				# 	stat_reg.append(filtered_mean_evec[idx])
				# 	stat_evec.append(toBinary(filtered_evec[idx]))

				# elif region_val == 0:
				# 	dyn_reg.append(filtered_mean_evec[idx])
				# 	dyn_evec.append(toBinary(filtered_evec[idx]))


		#save path
		if args.add_mean_evec:
			region_test_save_dir = os.path.join("data","test_by_region","{}_add_mean"\
				.format(args.resolution),cell)
		else:
			region_test_save_dir = os.path.join("data","test_by_region","{}"\
				.format(args.resolution),cell)

		if not os.path.exists(region_test_save_dir):
			os.makedirs(region_test_save_dir)
		print("save files to {}".format(region_test_save_dir))

		for region_val in region_data.keys():
			if region_val != float("NaN"):
				region_val_test_save_dir = os.path.join(region_test_save_dir,"region_{}".format(int(region_val)))
				if not os.path.exists(region_val_test_save_dir):
					os.makedirs(region_val_test_save_dir)
				
				with open(os.path.join(region_val_test_save_dir,"{}_{}_region{}_target.txt".format(cell,args.resolution,int(region_val))), "w") as outfile:
					outfile.write("\n".join(region_data[region_val]["evec"]))
				
				with open(os.path.join(region_val_test_save_dir,"{}_{}_region{}_predict.txt".format(cell,args.resolution,int(region_val))), "w") as outfile:
					outfile.write("\n".join(region_data[region_val]["mean_evec"]))

				if args.add_mean_evec:
					test_dataset = abDataset_with_mean(region_data[region_val]["test_data"])
					test_loader = DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False)
				else:
					test_dataset = abDataset(region_data[region_val]["test_data"])
					test_loader = DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False)
				torch.save(test_loader,os.path.join(region_val_test_save_dir,"{}_{}_region{}_test.pth".format(cell,args.resolution,int(region_val))))
	





		

def prepare_data_cross_validation_with_mean_evec_legacy(args,cell_lines):
	
	# cell_lines = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	# cell_lines = ["IMR90", "NHEK"]
	# read all data, store with key cellName_chrX: all data
	all_data = {}
	train_data_all = {}
	test_data_all = {}

	for cell_line in cell_lines:
		cell_line_dir = os.path.join(args.data_dir,cell_line)
		for filename in os.listdir(cell_line_dir):
			print("***Reading data from {}...".format(filename))
			chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as csvfile:
				data_reader = csv.reader(csvfile, delimiter=',')
				lines = []
				for row in data_reader:
					row = [val if val else 'NA' for val in row]
					lines.append(row)
				all_data[chromosome] = lines

	# compute mean value for each histone modification, replace NA with mean



	data_info = {0:{"sum":0, "count":0},\
				1:{"sum":0, "count":0},\
				2:{"sum":0, "count":0},\
				3:{"sum":0, "count":0},\
				4:{"sum":0, "count":0},\
				5:{"sum":0, "count":0}
				}
	empty_point = defaultdict(list)

	for chrom, data in all_data.items():
		for line_id, line in enumerate(data):
			for hm_idx, value in enumerate(line[3:]):
				# print(hm_idx)
				# print(value)
				if value == 'NA':
					empty_point[hm_idx].append((chrom,line_id))
				else:
					sum_value = data_info[hm_idx]["sum"] + float(value)
					data_count = data_info[hm_idx]["count"] + 1
					data_info[hm_idx]["sum"] = sum_value
					data_info[hm_idx]["count"] = data_count

	mean_values = {}
	print(data_info)
	for hm_idx, info in data_info.items():
		# print(hm_idx,info)
		mean_values[hm_idx] = info["sum"] / info["count"]

	# print(mean_values)

	for hm_idx, point_list in empty_point.items():
		mean_value = mean_values[hm_idx]
		for (chrom, line_id) in point_list:
			all_data[chrom][line_id][hm_idx+3] = mean_value

	print("***Finish Imputing***")

	#load mean eigenvector
	mean_evec = {}
	mean_evec_dir = os.path.join("data","mean_evec","mean_evec_updated_100kb_allchromosome_by_chromosome")

	
	for cell in cell_lines:
		for ch in range(1,23):
			load_path = os.path.join(mean_evec_dir,cell,"{}_chr{}_test.txt".format(cell,ch))
			f = open(load_path, "r")
			all_mean = f.readlines()
			all_mean = [mean.strip().lower() for mean in all_mean]
			
			#now load updated eigenvector
			updated_evec_file = os.path.join("data","updated_eigenvectors_100kb",cell,\
				"{}_eigen_100kb_chr{}.csv".format(cell,ch))
			f = open(updated_evec_file, "r")
			lines = f.readlines()
			eigenvectors = lines[1:]

			filter_means = []
			for idx, evec in enumerate(eigenvectors):
				evec = evec.split(',')[1].strip().lower()
				if evec != "":
					filter_means.append(all_mean[idx])

			mean_evec["{}_chr{}".format(cell,ch)] = filter_means
	#clean data
	cleaned_data = {}
	for key, item in all_data.items():
		cell = key.split("_")[0]
		chr_num = key.split("_")[1]
		if args.use_corre:
			corre = correlations[cell][chr_num]
		else:
			corre = None
		curr_mean_evec = mean_evec[key]
		cleaned_data[key] = clean_data_with_mean(item,corre,curr_mean_evec)
	#divide data into train, validate, testing	
	train_chrom = {}
	validate_chrom = {}
	test_chrom = {}

	

	for chrom, data in cleaned_data.items():
		# print(chrom)
		cell = chrom.split("_")[0]
		chr_num = chrom.split("_")[1][3:]

		if chr_num == "X":
			continue
		chr_num	 = int(chr_num)	
		if cell == args.cell:
			test_chrom[chrom] = data
		else:
			train_chrom[chrom] = data

	

	print("***Finish Cleaning***")
	
	
	# print(test_chrom.keys())
	#combine data
	training_data = combine_data(train_chrom)
	testing_data = combine_data(test_chrom)
	print("***Finish Combining***")

	#save test loader
	test_dataset = abDataset(testing_data)
	test_loader = DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False)
	torch.save(test_loader,os.path.join(args.data_save_dir,"test.pth"))
	


	# train_dataset = abDataset(training_data)
	# valid_dataset = abDataset(validation_data)
	# train_loader = DataLoader(train_dataset,batch_size=args.train_batch_size, shuffle=True)
	# valid_loader = DataLoader(valid_dataset,batch_size=args.valid_batch_size, shuffle=False)
	


	#spit data for cross-validation
	kfold =KFold(n_splits=args.num_fold)
	for fold, (train_index, valid_index) in enumerate(kfold.split(training_data)):
		print("TRAIN:", train_index, "TEST:", valid_index)
		train_fold = [training_data[i] for i in train_index]
		valid_fold = [training_data[i] for i in valid_index]

		# print(train_fold[0].keys())

		train_dataset = abDataset_with_mean(train_fold)
		valid_dataset = abDataset_with_mean(valid_fold)

		train_loader = DataLoader(train_dataset,batch_size=args.train_batch_size, shuffle=True)
		valid_loader = DataLoader(valid_dataset,batch_size=args.valid_batch_size, shuffle=False)

		fold_save_path = os.path.join(args.data_save_dir,str(fold))
		torch.save(train_loader,os.path.join(fold_save_path,"train.pth"))
		torch.save(valid_loader,os.path.join(fold_save_path,"valid.pth"))

	return testing_data


def prepare_data_cross_validation_with_mean_evec(args,cell_lines):
	
	# cell_lines = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	# cell_lines = ["IMR90", "HUVEC"]
	# read all data, store with key cellName_chrX: all data
	all_data = {}
	train_data_all = {}
	test_data_all = {}

	for cell_line in cell_lines:
		cell_line_dir = os.path.join(args.data_dir,cell_line)
		for filename in os.listdir(cell_line_dir):
			print("***Reading data from {}...".format(filename))
			chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as csvfile:
				data_reader = csv.reader(csvfile, delimiter=',')
				lines = []
				for row in data_reader:
					row = [val if val else 'NA' for val in row]
					lines.append(row)
				all_data[chromosome] = lines

	# compute mean value for each histone modification, replace NA with mean

	data_info = {0:{"sum":0, "count":0},\
				1:{"sum":0, "count":0},\
				2:{"sum":0, "count":0},\
				3:{"sum":0, "count":0},\
				4:{"sum":0, "count":0},\
				5:{"sum":0, "count":0}
				}
	empty_point = defaultdict(list)

	for chrom, data in all_data.items():
		for line_id, line in enumerate(data):
			for hm_idx, value in enumerate(line[3:]):
				# print(hm_idx)
				# print(value)
				if value == 'NA':
					empty_point[hm_idx].append((chrom,line_id))
				else:
					sum_value = data_info[hm_idx]["sum"] + float(value)
					data_count = data_info[hm_idx]["count"] + 1
					data_info[hm_idx]["sum"] = sum_value
					data_info[hm_idx]["count"] = data_count

	mean_values = {}
	print(data_info)
	for hm_idx, info in data_info.items():
		# print(hm_idx,info)
		mean_values[hm_idx] = info["sum"] / info["count"]

	# print(mean_values)

	for hm_idx, point_list in empty_point.items():
		mean_value = mean_values[hm_idx]
		for (chrom, line_id) in point_list:
			all_data[chrom][line_id][hm_idx+3] = mean_value

	print("***Finish Imputing***")

	#load mean eigenvector
	mean_evec = {}
	mean_evec_dir = os.path.join("data","mean_evec","mean_evec_updated_{}_allchromosome_by_chromosome".format(args.resolution))

	
	for cell in cell_lines:
		for ch in range(1,23):
			load_path = os.path.join(mean_evec_dir,cell,"{}_chr{}_test.txt".format(cell,ch))
			# print("load mean evec from {}...".format(load_path))
			f = open(load_path, "r")
			read_mean = f.readlines()
			all_mean = [mean.strip().lower() for mean in read_mean]

			updated_evec_file = os.path.join("data","updated_eigenvectors_{}".format(args.resolution),cell,\
				"{}_eigen_{}_chr{}.csv".format(cell,args.resolution,ch))
			f = open(updated_evec_file, "r")
			lines = f.readlines()
			eigenvectors = lines[1:]
			eigenvectors = [evec.split(',')[1].strip().lower() for evec in eigenvectors]
			
			filter_means = []

			count = 0

			for idx, evec in enumerate(eigenvectors):
				if evec == "":
					continue
				else:
					filter_means.append(all_mean[idx])
			# print("{} {}: {}".format(cell,ch,count))

			mean_evec["{}_chr{}".format(cell,ch)] = filter_means
	# clean data
	# exit(0)
	cleaned_data = {}
	for key, item in all_data.items():
		cell = key.split("_")[0]
		chr_num = key.split("_")[1]
		if args.use_corre:
			corre = correlations[cell][chr_num]
		else:
			corre = None
		curr_mean_evec = mean_evec[key]
		# print(key)
		cleaned_data[key] = clean_data_with_mean(item,corre,curr_mean_evec)
	# exit()
	#divide data into train, validate, testing	
	train_chrom = {}
	validate_chrom = {}
	test_chrom = {}

	

	for chrom, data in cleaned_data.items():
		# print(chrom)
		cell = chrom.split("_")[0]
		chr_num = chrom.split("_")[1][3:]

		if chr_num == "X":
			continue
		chr_num	 = int(chr_num)	
		if cell == args.cell:
			test_chrom[chrom] = data
		else:
			train_chrom[chrom] = data


	print("***Finish Cleaning***")
	
	# print(test_chrom.keys())
	#combine data
	training_data = combine_data(train_chrom)
	testing_data = combine_data(test_chrom)
	print("***Finish Combining***")

	#save test loader
	test_dataset = abDataset_with_mean(testing_data)
	test_loader = DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False)
	torch.save(test_loader,os.path.join(args.data_save_dir,"test.pth"))
	

	#spit data for cross-validation
	kfold =KFold(n_splits=args.num_fold)
	for fold, (train_index, valid_index) in enumerate(kfold.split(training_data)):
		print("TRAIN:", train_index, "TEST:", valid_index)
		train_fold = [training_data[i] for i in train_index]
		valid_fold = [training_data[i] for i in valid_index]

		# print(train_fold[0].keys())

		train_dataset = abDataset_with_mean(train_fold)
		valid_dataset = abDataset_with_mean(valid_fold)

		train_loader = DataLoader(train_dataset,batch_size=args.train_batch_size, shuffle=True)
		valid_loader = DataLoader(valid_dataset,batch_size=args.valid_batch_size, shuffle=False)

		fold_save_path = os.path.join(args.data_save_dir,str(fold))
		torch.save(train_loader,os.path.join(fold_save_path,"train.pth"))
		torch.save(valid_loader,os.path.join(fold_save_path,"valid.pth"))

	return testing_data

def prepare_data_cross_validation_with_mean_evec_updated(args,cell_lines):
	
	# cell_lines = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	# cell_lines = ["IMR90", "HUVEC"]
	# read all data, store with key cellName_chrX: all data
	all_data = {}
	train_data_all = {}
	test_data_all = {}

	for cell_line in cell_lines:
		cell_line_dir = os.path.join(args.data_dir,cell_line)
		for filename in os.listdir(cell_line_dir):
			print("***Reading data from {}...".format(filename))
			chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as csvfile:
				data_reader = csv.reader(csvfile, delimiter=',')
				lines = []
				for row in data_reader:
					row = [val if val else 'NA' for val in row]
					lines.append(row)
				all_data[chromosome] = lines

	# compute mean value for each histone modification, replace NA with mean

	data_info = {0:{"sum":0, "count":0},\
				1:{"sum":0, "count":0},\
				2:{"sum":0, "count":0},\
				3:{"sum":0, "count":0},\
				4:{"sum":0, "count":0},\
				5:{"sum":0, "count":0}
				}
	empty_point = defaultdict(list)

	for chrom, data in all_data.items():
		for line_id, line in enumerate(data):
			for hm_idx, value in enumerate(line[3:]):
				# print(hm_idx)
				# print(value)
				if value == 'NA':
					empty_point[hm_idx].append((chrom,line_id))
				else:
					sum_value = data_info[hm_idx]["sum"] + float(value)
					data_count = data_info[hm_idx]["count"] + 1
					data_info[hm_idx]["sum"] = sum_value
					data_info[hm_idx]["count"] = data_count

	mean_values = {}
	print(data_info)
	for hm_idx, info in data_info.items():
		# print(hm_idx,info)
		mean_values[hm_idx] = info["sum"] / info["count"]

	# print(mean_values)

	for hm_idx, point_list in empty_point.items():
		mean_value = mean_values[hm_idx]
		for (chrom, line_id) in point_list:
			all_data[chrom][line_id][hm_idx+3] = mean_value

	print("***Finish Imputing***")

	#load mean eigenvector
	mean_evec = {}
	mean_evec_dir = os.path.join("data","mean_evec","mean_evec_updated_{}_allchromosome_by_chromosome".format(args.resolution))


	all_evec = {}
	
	for cell in cell_lines:
		cell_evecs = []
		for ch in range(1,23):
			# in this setting, the mean eigenvector is always the mean of 5 cells exclude the target cell
			load_path = os.path.join(mean_evec_dir,args.cell,"{}_chr{}_test.txt".format(args.cell,ch))
			# print("load mean evec from {}...".format(load_path))
			f = open(load_path, "r")
			read_mean = f.readlines()
			all_mean = [mean.strip().lower() for mean in read_mean]

			updated_evec_file = os.path.join("data","updated_eigenvectors_{}".format(args.resolution),cell,\
				"{}_eigen_{}_chr{}.csv".format(cell,args.resolution,ch))
			f = open(updated_evec_file, "r")
			lines = f.readlines()
			eigenvectors = lines[1:]
			eigenvectors = [evec.split(',')[1].strip().lower() for evec in eigenvectors]

			evec_np = np.array([np.nan if line=="" else np.float(line) for line in eigenvectors])

			if len(cell_evecs)==0:
				cell_evecs = evec_np
			else:
				cell_evecs = np.concatenate((cell_evecs,evec_np))

			
			filter_means = []

			count = 0

			for idx, evec in enumerate(eigenvectors):
				if evec == "":
					continue
				else:
					filter_means.append(all_mean[idx])

			mean_evec["{}_chr{}".format(cell,ch)] = filter_means

		all_evec[cell] = cell_evecs

	cleaned_data = {}
	

	for key, item in all_data.items():
		cell = key.split("_")[0]
		chr_num = key.split("_")[1]
		if args.use_corre:
			corre = correlations[cell][chr_num]
		else:
			corre = None
		
		curr_mean_evec = mean_evec[key]
		# print(key)
		cleaned_data[key] = clean_data_with_mean(item,corre,curr_mean_evec)
	# exit()
	#divide data into train, validate, testing	
	train_chrom = {}
	validate_chrom = {}
	test_chrom = {}

	

	for chrom, data in cleaned_data.items():
		# print(chrom)
		cell = chrom.split("_")[0]
		chr_num = chrom.split("_")[1][3:]

		if chr_num == "X":
			continue
		chr_num	 = int(chr_num)	
		if cell == args.cell:
			test_chrom[chrom] = data
		else:
			train_chrom[chrom] = data


	print("***Finish Cleaning***")
	
	# print(test_chrom.keys())
	#combine data
	training_data = combine_data(train_chrom)
	testing_data = combine_data(test_chrom)
	print("***Finish Combining***")

	#save test loader
	test_dataset = abDataset_with_mean(testing_data)
	test_loader = DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False)
	torch.save(test_loader,os.path.join(args.data_save_dir,"test.pth"))


	cell_lines.remove(args.cell)
	
	for fold_idx, cell in enumerate(cell_lines):
		print("saving train and valid data for fold {}, valid cell: {}".format(fold_idx,cell))
		train_fold = {}
		valid_fold = {}
		
		valid_cell = cell

		for chrom, data in train_chrom.items(): 
			cell = chrom.split("_")[0]
			chr_num = chrom.split("_")[1][3:]

			if cell == valid_cell:
				valid_fold[chrom] = data
			else:
				train_fold[chrom] = data

		training_data = combine_data(train_fold)
		validation_data = combine_data(valid_fold)

		train_dataset = abDataset_with_mean(training_data)
		valid_dataset = abDataset_with_mean(validation_data)

		train_loader = DataLoader(train_dataset,batch_size=args.train_batch_size, shuffle=True)
		valid_loader = DataLoader(valid_dataset,batch_size=args.valid_batch_size, shuffle=False)

		fold_save_path = os.path.join(args.data_save_dir,str(valid_cell))
		torch.save(train_loader,os.path.join(fold_save_path,"train.pth"))
		torch.save(valid_loader,os.path.join(fold_save_path,"valid.pth"))

	return testing_data

def prepare_test_data_with_mean_evec_strong_compartment(args,cell_lines):
	
	# cell_lines = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	# cell_lines = ["IMR90", "HUVEC"]
	# read all data, store with key cellName_chrX: all data
	print("get strong compartments")

	all_data = {}
	train_data_all = {}
	test_data_all = {}
	for cell_line in cell_lines:
		cell_line_dir = os.path.join(args.data_dir,cell_line)
		for filename in os.listdir(cell_line_dir):
			print("***Reading data from {}...".format(filename))
			chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as csvfile:
				data_reader = csv.reader(csvfile, delimiter=',')
				lines = []
				for row in data_reader:
					row = [val if val else 'NA' for val in row]
					lines.append(row)
				all_data[chromosome] = lines

	# compute mean value for each histone modification, replace NA with mean

	data_info = {0:{"sum":0, "count":0},\
				1:{"sum":0, "count":0},\
				2:{"sum":0, "count":0},\
				3:{"sum":0, "count":0},\
				4:{"sum":0, "count":0},\
				5:{"sum":0, "count":0}
				}
	empty_point = defaultdict(list)

	for chrom, data in all_data.items():
		for line_id, line in enumerate(data):
			for hm_idx, value in enumerate(line[3:]):
				# print(hm_idx)
				# print(value)
				if value == 'NA':
					empty_point[hm_idx].append((chrom,line_id))
				else:
					sum_value = data_info[hm_idx]["sum"] + float(value)
					data_count = data_info[hm_idx]["count"] + 1
					data_info[hm_idx]["sum"] = sum_value
					data_info[hm_idx]["count"] = data_count

	mean_values = {}
	print(data_info)
	for hm_idx, info in data_info.items():
		# print(hm_idx,info)
		mean_values[hm_idx] = info["sum"] / info["count"]

	# print(mean_values)

	for hm_idx, point_list in empty_point.items():
		mean_value = mean_values[hm_idx]
		for (chrom, line_id) in point_list:
			all_data[chrom][line_id][hm_idx+3] = mean_value

	print("***Finish Imputing***")

	#load mean eigenvector
	mean_evec = {}
	mean_evec_dir = os.path.join("data","mean_evec","mean_evec_updated_{}_allchromosome_by_chromosome".format(args.resolution))


	all_evec = {}
	
	for cell in cell_lines:
		cell_evecs = []
		for ch in range(1,23):
			# in this setting, the mean eigenvector is always the mean of 5 cells exclude the target cell
			load_path = os.path.join(mean_evec_dir,cell,"{}_chr{}_test.txt".format(cell,ch))
			# print("load mean evec from {}...".format(load_path))
			f = open(load_path, "r")
			read_mean = f.readlines()
			all_mean = [mean.strip().lower() for mean in read_mean]

			updated_evec_file = os.path.join("data","updated_eigenvectors_{}".format(args.resolution),cell,\
				"{}_eigen_{}_chr{}.csv".format(cell,args.resolution,ch))
			f = open(updated_evec_file, "r")
			lines = f.readlines()
			eigenvectors = lines[1:]
			eigenvectors = [evec.split(',')[1].strip().lower() for evec in eigenvectors]

			evec_np = np.array([np.nan if line=="" else np.float(line) for line in eigenvectors])

			if len(cell_evecs)==0:
				cell_evecs = evec_np
			else:
				cell_evecs = np.concatenate((cell_evecs,evec_np))

			filter_means = []
			count = 0
			for idx, evec in enumerate(eigenvectors):
				if evec == "":
					continue
				else:
					filter_means.append(all_mean[idx])

			mean_evec["{}_chr{}".format(cell,ch)] = filter_means

		all_evec[cell] = cell_evecs

	cleaned_data = {}
	

	for key, item in all_data.items():
		cell = key.split("_")[0]
		chr_num = key.split("_")[1]
		if args.use_corre:
			corre = correlations[cell][chr_num]
		else:
			corre = None
		curr_mean_evec = mean_evec[key]
		# print(key)
		cleaned_data[key] = clean_data_with_mean_and_evec(item,corre,curr_mean_evec)
	# exit()
	#divide data into train, validate, testing	
	print("***Finish Cleaning***")

	#Get mean and std for A/B compartment
	data_values = list(cleaned_data.values())
	# print(data_values[0])
	all_evec = []
	for i_list in data_values:
		for item in i_list:
			all_evec.append(item["reg_label"])

	all_labels = []
	for i_list in data_values:
		for item in i_list:
			all_labels.append(item["cla_label"])
	
	zero_count = all_labels.count(0)
	print("0 count: ",zero_count)
	
	all_A = [] #[item for item in all_evec if item > 0]
	all_B = []#[item for item in all_evec if item < 0]
	
	for item in all_evec:
		if item>0:
			all_A.append(item)
		else:
			all_B.append(item)
	print(all_A[:10])
	print(all_B[:10])
	A_mean = np.mean(all_A)
	A_std = np.std(all_A)
	B_mean = np.mean(all_B)
	B_std = np.std(all_B)
	print("A: mean = {}, std = {}".format(A_mean,A_std))
	print("B: mean = {}, std = {}".format(B_mean,B_std))
	#
	strong_data = {}
	removed_data = 0 
	for chrom, data in cleaned_data.items():
		new_data = []
		for item in data:
			#"data""evec""cla_label""reg_label""mean_evec"
			if item["cla_label"] == 1:
				if item["reg_label"] > A_mean - A_std:
					new_data.append(item)
				else:
					removed_data += 1
			else:
				if item["reg_label"] < B_mean + B_std:
					new_data.append(item)
				else:
					removed_data += 1
		strong_data[chrom] = new_data
	print("====================================")
	print("removed: ", removed_data)
	print("====================================")
	test = {}
	for cell in cell_lines:
		test[cell] = {}
		for ch in range(1,23):
			chrom = "{}_chr{}".format(cell,ch)
			item = strong_data[chrom]
			test[cell][chrom] = item
	
	for cell in cell_lines:
		combined = combine_data(test[cell])
		test_dataset = abDataset_with_mean(combined)
		test_loader = DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False)

		cell_path = os.path.join(args.data_save_dir,cell)
		os.makedirs(cell_path)
		torch.save(test_loader,os.path.join(cell_path,"test.pth"))
	
def prepare_data_cross_validation_with_mean_evec_updated_allcells(args,cell_lines):
	
	# cell_lines = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	# cell_lines = ["IMR90", "HUVEC"]
	# read all data, store with key cellName_chrX: all data
	all_data = {}
	train_data_all = {}
	test_data_all = {}

	for cell_line in cell_lines:
		cell_line_dir = os.path.join(args.data_dir,cell_line)
		for filename in os.listdir(cell_line_dir):
			print("***Reading data from {}...".format(filename))
			chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as csvfile:
				data_reader = csv.reader(csvfile, delimiter=',')
				lines = []
				for row in data_reader:
					row = [val if val else 'NA' for val in row]
					lines.append(row)
				all_data[chromosome] = lines

	# compute mean value for each histone modification, replace NA with mean

	data_info = {0:{"sum":0, "count":0},\
				1:{"sum":0, "count":0},\
				2:{"sum":0, "count":0},\
				3:{"sum":0, "count":0},\
				4:{"sum":0, "count":0},\
				5:{"sum":0, "count":0}
				}
	empty_point = defaultdict(list)
	for chrom, data in all_data.items():
		for line_id, line in enumerate(data):
			for hm_idx, value in enumerate(line[3:]):
				if value == 'NA':
					empty_point[hm_idx].append((chrom,line_id))
				else:
					sum_value = data_info[hm_idx]["sum"] + float(value)
					data_count = data_info[hm_idx]["count"] + 1
					data_info[hm_idx]["sum"] = sum_value
					data_info[hm_idx]["count"] = data_count

	mean_values = {}
	print(data_info)
	for hm_idx, info in data_info.items():
		# print(hm_idx,info)
		mean_values[hm_idx] = info["sum"] / info["count"]

	# print(mean_values)

	for hm_idx, point_list in empty_point.items():
		mean_value = mean_values[hm_idx]
		for (chrom, line_id) in point_list:
			all_data[chrom][line_id][hm_idx+3] = mean_value

	print("***Finish Imputing***")

	mean_evec_dir = os.path.join("data","mean_evec","mean_evec_updated_{}_allchromosome_by_chromosome".format(args.resolution))

	#read all means:
	all_meanevec = {}
	for cell in cell_lines:
		for ch in range(1,23):
			load_path = os.path.join(mean_evec_dir,cell,"{}_chr{}_test.txt".format(cell,ch))
				# print("load mean evec from {}...".format(load_path))
			f = open(load_path, "r")
			read_mean = f.readlines()
			mean = [mean.strip().lower() for mean in read_mean]
			all_meanevec["{}_chr{}".format(cell,ch)] = mean


	#load mean eigenvector
	mean_evec = {}
	

	for fold in range(args.num_fold):
		all_evec = {}
		valid_cell = cell_lines[fold]
		for cell in cell_lines:
			cell_evecs = []
			for ch in range(1,23):
				# in this setting, the mean eigenvector is the mean of 5 cells exclude the validate cell
				
				all_mean = all_meanevec["{}_chr{}".format(valid_cell,ch)]

				updated_evec_file = os.path.join("data","updated_eigenvectors_{}".format(args.resolution),cell,\
					"{}_eigen_{}_chr{}.csv".format(cell,args.resolution,ch))
				f = open(updated_evec_file, "r")
				lines = f.readlines()
				eigenvectors = lines[1:]
				eigenvectors = [evec.split(',')[1].strip().lower() for evec in eigenvectors]

				evec_np = np.array([np.nan if line=="" else np.float(line) for line in eigenvectors])

				if len(cell_evecs)==0:
					cell_evecs = evec_np
				else:
					cell_evecs = np.concatenate((cell_evecs,evec_np))

				
				filter_means = []

				count = 0
				for idx, evec in enumerate(eigenvectors):
					if evec == "":
						continue
					else:
						filter_means.append(all_mean[idx])

				mean_evec["{}_chr{}".format(cell,ch)] = filter_means

			all_evec[cell] = cell_evecs

		cleaned_data = {}
		for key, item in all_data.items():
			cell = key.split("_")[0]
			chr_num = key.split("_")[1]
			if args.use_corre:
				corre = correlations[cell][chr_num]
			else:
				corre = None
			curr_mean_evec = mean_evec[key]
			# print(key)
			cleaned_data[key] = clean_data_with_mean(item,corre,curr_mean_evec)
		# exit()
		#divide data into train, validate, testing	
		print("***Finish Cleaning***")


		train_chrom = {}
		validate_chrom = {}
		test_chrom = {}

		for chrom, data in cleaned_data.items():
			# print(chrom)
			cell = chrom.split("_")[0]
			chr_num = chrom.split("_")[1][3:]
			if chr_num == "X":
				continue
			chr_num	 = int(chr_num)	
			if cell == valid_cell:
				validate_chrom[chrom] = data
			else:
				train_chrom[chrom] = data

		# print(test_chrom.keys())
		#combine data
		training_data = combine_data(train_chrom)
		validation_data = combine_data(validate_chrom)
		print("***Finish Combining***")


		# cell_lines.remove(args.cell)
	
		
		print("saving train and valid data for fold {}, valid cell: {}".format(fold,valid_cell))
		# train_fold = {}
		# valid_fold = {}
		
		# valid_cell = cell

		# for chrom, data in train_chrom.items(): 
		# 	cell = chrom.split("_")[0]
		# 	chr_num = chrom.split("_")[1][3:]

		# 	if cell == valid_cell:
		# 		valid_fold[chrom] = data
		# 	else:
		# 		train_fold[chrom] = data

		# training_data = combine_data(train_fold)
		# validation_data = combine_data(valid_fold)

		train_dataset = abDataset_with_mean(training_data)
		valid_dataset = abDataset_with_mean(validation_data)

		train_loader = DataLoader(train_dataset,batch_size=args.train_batch_size, shuffle=True)
		valid_loader = DataLoader(valid_dataset,batch_size=args.valid_batch_size, shuffle=False)

		fold_save_path = os.path.join(args.data_save_dir,str(valid_cell))
		torch.save(train_loader,os.path.join(fold_save_path,"train.pth"))
		torch.save(valid_loader,os.path.join(fold_save_path,"valid.pth"))

def prepare_data_for_test_cell(args):

	# cell_lines = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	# cell_lines = ["IMR90", "HUVEC"]
	# read all data, store with key cellName_chrX: all data
	all_data = {}
	train_data_all = {}
	test_data_all = {}
	cell_lines = [args.cell]

	for cell_line in cell_lines:
		cell_line_dir = os.path.join(args.data_dir,cell_line)
		for filename in os.listdir(cell_line_dir):
			print("***Reading data from {}...".format(filename))
			chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as csvfile:
				data_reader = csv.reader(csvfile, delimiter=',')
				lines = []
				for row in data_reader:
					row = [val if val else 'NA' for val in row]
					lines.append(row)
				all_data[chromosome] = lines

	# compute mean value for each histone modification, replace NA with mean

	data_info = {0:{"sum":0, "count":0},\
				1:{"sum":0, "count":0},\
				2:{"sum":0, "count":0},\
				3:{"sum":0, "count":0},\
				4:{"sum":0, "count":0},\
				5:{"sum":0, "count":0}
				}
	empty_point = defaultdict(list)

	for chrom, data in all_data.items():
		for line_id, line in enumerate(data):
			for hm_idx, value in enumerate(line[3:]):
				# print(hm_idx)
				# print(value)
				if value == 'NA':
					empty_point[hm_idx].append((chrom,line_id))
				else:
					sum_value = data_info[hm_idx]["sum"] + float(value)
					data_count = data_info[hm_idx]["count"] + 1
					data_info[hm_idx]["sum"] = sum_value
					data_info[hm_idx]["count"] = data_count

	mean_values = {}
	print(data_info)
	for hm_idx, info in data_info.items():
		# print(hm_idx,info)
		mean_values[hm_idx] = info["sum"] / info["count"]

	# print(mean_values)

	for hm_idx, point_list in empty_point.items():
		mean_value = mean_values[hm_idx]
		for (chrom, line_id) in point_list:
			all_data[chrom][line_id][hm_idx+3] = mean_value

	print("***Finish Imputing***")

	#load mean eigenvector
	mean_evec = {}
	mean_evec_dir = os.path.join("data","mean_evec","mean_evec_updated_{}_allchromosome_by_chromosome".format(args.resolution))


	all_evec = {}
	
	for cell in cell_lines:
		cell_evecs = []
		for ch in range(1,22):
			# in this setting, the mean eigenvector is always the mean of 5 cells exclude the target cell
			load_path = os.path.join(mean_evec_dir,args.cell,"{}_chr{}_test.txt".format(args.cell,ch))
			# print("load mean evec from {}...".format(load_path))
			try:
				f = open(load_path, "r")
			except:
				print("cannot find file",load_path)
				continue
			read_mean = f.readlines()
			all_mean = [mean.strip().lower() for mean in read_mean]

			updated_evec_file = os.path.join("data","updated_eigenvectors_{}".format(args.resolution),cell,\
				"{}_eigen_{}_chr{}.csv".format(cell,args.resolution,ch))
			f = open(updated_evec_file, "r")
			lines = f.readlines()
			eigenvectors = lines[1:]
			eigenvectors = [evec.split(',')[1].strip().lower() for evec in eigenvectors]

			evec_np = np.array([np.nan if line=="" else np.float(line) for line in eigenvectors])

			if len(cell_evecs)==0:
				cell_evecs = evec_np
			else:
				cell_evecs = np.concatenate((cell_evecs,evec_np))

			filter_means = []

			count = 0

			for idx, evec in enumerate(eigenvectors):
				if evec == "":
					continue
				else:
					filter_means.append(all_mean[idx])

			mean_evec["{}_chr{}".format(cell,ch)] = filter_means

		all_evec[cell] = cell_evecs

	cleaned_data = {}
	
	print(mean_evec.keys())
	print(all_evec.keys())
	for key, item in all_data.items():
		print(key)
		if key == 'GM23248_chr22':
			continue
		cell = key.split("_")[0]
		chr_num = key.split("_")[1]
		if args.use_corre:
			corre = correlations[cell][chr_num]
		else:
			corre = None
		
		curr_mean_evec = mean_evec[key]
		# print(key)
		cleaned_data[key] = clean_data_with_mean(item,corre,curr_mean_evec)
	# exit()
	#divide data into train, validate, testing	
	train_chrom = {}
	validate_chrom = {}
	test_chrom = {}

	

	for chrom, data in cleaned_data.items():
		# print(chrom)
		cell = chrom.split("_")[0]
		chr_num = chrom.split("_")[1][3:]

		if chr_num == "X":
			continue
		chr_num	 = int(chr_num)	
		if cell == args.cell:
			test_chrom[chrom] = data
		else:
			train_chrom[chrom] = data


	print("***Finish Cleaning***")
	
	# print(test_chrom.keys())
	#combine data
	training_data = combine_data(train_chrom)
	testing_data = combine_data(test_chrom)
	print("***Finish Combining***")

	#save test loader
	test_dataset = abDataset_with_mean(testing_data)
	test_loader = DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False)
	torch.save(test_loader,os.path.join(args.data_save_dir,"test.pth"))


def prepare_data_cross_validation_with_mean_evec_updated_add_region(args,cell_lines):
	
	# cell_lines = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	# cell_lines = ["IMR90", "HUVEC"]
	# read all data, store with key cellName_chrX: all data
	all_data = {}
	train_data_all = {}
	test_data_all = {}

	for cell_line in cell_lines:
		cell_line_dir = os.path.join(args.data_dir,cell_line)
		for filename in os.listdir(cell_line_dir):
			print("***Reading data from {}...".format(filename))
			chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as csvfile:
				data_reader = csv.reader(csvfile, delimiter=',')
				lines = []
				for row in data_reader:
					row = [val if val else 'NA' for val in row]
					lines.append(row)
				all_data[chromosome] = lines

	# compute mean value for each histone modification, replace NA with mean

	data_info = {0:{"sum":0, "count":0},\
				1:{"sum":0, "count":0},\
				2:{"sum":0, "count":0},\
				3:{"sum":0, "count":0},\
				4:{"sum":0, "count":0},\
				5:{"sum":0, "count":0}
				}
	empty_point = defaultdict(list)

	for chrom, data in all_data.items():
		for line_id, line in enumerate(data):
			for hm_idx, value in enumerate(line[3:]):
				# print(hm_idx)
				# print(value)
				if value == 'NA':
					empty_point[hm_idx].append((chrom,line_id))
				else:
					sum_value = data_info[hm_idx]["sum"] + float(value)
					data_count = data_info[hm_idx]["count"] + 1
					data_info[hm_idx]["sum"] = sum_value
					data_info[hm_idx]["count"] = data_count

	mean_values = {}
	print(data_info)
	for hm_idx, info in data_info.items():
		# print(hm_idx,info)
		mean_values[hm_idx] = info["sum"] / info["count"]

	# print(mean_values)

	for hm_idx, point_list in empty_point.items():
		mean_value = mean_values[hm_idx]
		for (chrom, line_id) in point_list:
			all_data[chrom][line_id][hm_idx+3] = mean_value

	print("***Finish Imputing***")

	
	mean_evec_dir = os.path.join("data","mean_evec","mean_evec_updated_{}_allchromosome_by_chromosome".format(args.resolution))
	region_dir = os.path.join("data","region","region_{}_by_chromosome".format(args.resolution))
	updated_evec_dir = os.path.join("data","updated_eigenvectors_{}".format(args.resolution))


	mean_evec = {}
	region_data = {}

	#process mean and region file based on the eigenvectors files

	print("***Load mean and regions***")
	
	for cell in cell_lines:
		for ch in range(1,23):
			# in this setting, the mean eigenvector is always the mean of 5 cells exclude the target cell

			print("{}: chr {}".format(cell,ch))

			#load region file
			region_file_path = os.path.join(region_dir,cell,"region_{}_chr{}.txt".format(cell,chr_num))
			region_f = open(region_file_path,"r")
			region_lines = region_f.readlines()
			region_lines = [float(line.strip().lower()) for line in region_lines]


			#load mean eigenvector
			load_path = os.path.join(mean_evec_dir,args.cell,"{}_chr{}_test.txt".format(args.cell,ch))
			f = open(load_path, "r")
			read_mean = f.readlines()
			all_mean = [mean.strip().lower() for mean in read_mean]

			#load eigenvectors

			updated_evec_file = os.path.join("data","updated_eigenvectors_{}".format(args.resolution),cell,\
				"{}_eigen_{}_chr{}.csv".format(cell,args.resolution,ch))
			f = open(updated_evec_file, "r")
			lines = f.readlines()
			eigenvectors = lines[1:]
			eigenvectors = [evec.split(',')[1].strip().lower() for evec in eigenvectors]
			evec_np = np.array([np.nan if line=="" else np.float(line) for line in eigenvectors])

			filter_means = []
			filter_region = []
			count = 0

			for idx, evec in enumerate(eigenvectors):
				if evec == "":
					continue
				else:
					filter_means.append(all_mean[idx])
					filter_region.append(region_lines[idx])

			mean_evec["{}_chr{}".format(cell,ch)] = filter_means
			region_data["{}_chr{}".format(cell,ch)] = filter_region

	cleaned_data = {}
	

	for key, item in all_data.items():
		cell = key.split("_")[0]
		chr_num = key.split("_")[1]
		if args.use_corre:
			corre = correlations[cell][chr_num]
		else:
			corre = None
		
		curr_mean_evec = mean_evec[key]
		curr_region = region_data[key]
		# print(key)
		cleaned_data[key] = clean_data_with_mean_and_region(item,corre,curr_mean_evec,curr_region)
	# exit()
	#divide data into train, validate, testing	
	train_chrom = {}
	validate_chrom = {}
	test_chrom = {}

	

	for chrom, data in cleaned_data.items():
		# print(chrom)
		cell = chrom.split("_")[0]
		chr_num = chrom.split("_")[1][3:]

		if chr_num == "X":
			continue
		chr_num	 = int(chr_num)	
		if cell == args.cell:
			test_chrom[chrom] = data
		else:
			train_chrom[chrom] = data


	print("***Finish Cleaning***")
	
	# print(test_chrom.keys())
	#combine data
	training_data = combine_data(train_chrom)
	testing_data = combine_data(test_chrom)
	print("***Finish Combining***")

	#save test loader
	test_dataset = abDataset_with_mean(testing_data)
	test_loader = DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False)
	torch.save(test_loader,os.path.join(args.data_save_dir,"test.pth"))


	cell_lines.remove(args.cell)
	
	for fold_idx, cell in enumerate(cell_lines):
		print("saving train and valid data for fold {}, valid cell: {}".format(fold_idx,cell))
		train_fold = {}
		valid_fold = {}
		
		valid_cell = cell

		for chrom, data in train_chrom.items(): 
			cell = chrom.split("_")[0]
			chr_num = chrom.split("_")[1][3:]

			if cell == valid_cell:
				valid_fold[chrom] = data
			else:
				train_fold[chrom] = data

		training_data = combine_data(train_fold)
		validation_data = combine_data(valid_fold)

		train_dataset = abDataset_with_mean(training_data)
		valid_dataset = abDataset_with_mean(validation_data)

		train_loader = DataLoader(train_dataset,batch_size=args.train_batch_size, shuffle=True)
		valid_loader = DataLoader(valid_dataset,batch_size=args.valid_batch_size, shuffle=False)

		fold_save_path = os.path.join(args.data_save_dir,str(valid_cell))
		torch.save(train_loader,os.path.join(fold_save_path,"train.pth"))
		torch.save(valid_loader,os.path.join(fold_save_path,"valid.pth"))

	return testing_data

def prepare_data_random_forest(args,cell_lines):
	all_data = {}
	train_data_all = {}
	test_data_all = {}

	for cell_line in cell_lines:
		cell_line_dir = os.path.join(args.data_dir,cell_line)
		for filename in os.listdir(cell_line_dir):
			print("***Reading data from {}...".format(filename))
			chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as csvfile:
				data_reader = csv.reader(csvfile, delimiter=',')
				lines = []
				for row in data_reader:
					row = [val if val else 'NA' for val in row]
					lines.append(row)
				all_data[chromosome] = lines

	# compute mean value for each histone modification, replace NA with mean

	data_info = {0:{"sum":0, "count":0},\
				1:{"sum":0, "count":0},\
				2:{"sum":0, "count":0},\
				3:{"sum":0, "count":0},\
				4:{"sum":0, "count":0},\
				5:{"sum":0, "count":0}
				}
	empty_point = defaultdict(list)

	for chrom, data in all_data.items():
		for line_id, line in enumerate(data):
			for hm_idx, value in enumerate(line[3:]):
				if value == 'NA':
					empty_point[hm_idx].append((chrom,line_id))
				else:
					sum_value = data_info[hm_idx]["sum"] + float(value)
					data_count = data_info[hm_idx]["count"] + 1
					data_info[hm_idx]["sum"] = sum_value
					data_info[hm_idx]["count"] = data_count

	mean_values = {}
	print(data_info)
	for hm_idx, info in data_info.items():
		# print(hm_idx,info)
		mean_values[hm_idx] = info["sum"] / info["count"]

	# print(mean_values)

	for hm_idx, point_list in empty_point.items():
		mean_value = mean_values[hm_idx]
		for (chrom, line_id) in point_list:
			all_data[chrom][line_id][hm_idx+3] = mean_value

	print("***Finish Imputing***")

	#clean data
	cleaned_data = {}
	for key, item in all_data.items():
		cell = key.split("_")[0]
		chr_num = key.split("_")[1]
		cleaned_data[key] = clean_data(item,None)


	return cleaned_data


def prepare_data_random_forest_with_mean_evec(args,cell_lines):
	print("prepare_data_random_forest_with_mean_evec")
	
	# cell_lines = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	# cell_lines = ["IMR90", "HUVEC"]
	# read all data, store with key cellName_chrX: all data
	all_data = {}
	train_data_all = {}
	test_data_all = {}

	for cell_line in cell_lines:
		cell_line_dir = os.path.join(args.data_dir,cell_line)
		for filename in os.listdir(cell_line_dir):
			print("***Reading data from {}...".format(filename))
			chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as csvfile:
				data_reader = csv.reader(csvfile, delimiter=',')
				lines = []
				for row in data_reader:
					row = [val if val else 'NA' for val in row]
					lines.append(row)
				all_data[chromosome] = lines

	# compute mean value for each histone modification, replace NA with mean

	data_info = {0:{"sum":0, "count":0},\
				1:{"sum":0, "count":0},\
				2:{"sum":0, "count":0},\
				3:{"sum":0, "count":0},\
				4:{"sum":0, "count":0},\
				5:{"sum":0, "count":0}
				}
	empty_point = defaultdict(list)

	for chrom, data in all_data.items():
		for line_id, line in enumerate(data):
			for hm_idx, value in enumerate(line[3:]):
				# print(hm_idx)
				# print(value)
				if value == 'NA':
					empty_point[hm_idx].append((chrom,line_id))
				else:
					sum_value = data_info[hm_idx]["sum"] + float(value)
					data_count = data_info[hm_idx]["count"] + 1
					data_info[hm_idx]["sum"] = sum_value
					data_info[hm_idx]["count"] = data_count

	mean_values = {}
	print(data_info)
	for hm_idx, info in data_info.items():
		# print(hm_idx,info)
		mean_values[hm_idx] = info["sum"] / info["count"]

	# print(mean_values)

	for hm_idx, point_list in empty_point.items():
		mean_value = mean_values[hm_idx]
		for (chrom, line_id) in point_list:
			all_data[chrom][line_id][hm_idx+3] = mean_value

	print("***Finish Imputing***")

	#load mean eigenvector
	mean_evec = {}
	mean_evec_dir = os.path.join("data","mean_evec","mean_evec_updated_{}_allchromosome_by_chromosome".format(args.resolution))


	all_evec = {}
	
	for cell in cell_lines:
		cell_evecs = []
		for ch in range(1,23):
			# in this setting, the mean eigenvector is always the mean of 5 cells exclude the target cell
			load_path = os.path.join(mean_evec_dir,args.cell,"{}_chr{}_test.txt".format(args.cell,ch))
			# print("load mean evec from {}...".format(load_path))
			f = open(load_path, "r")
			read_mean = f.readlines()
			all_mean = [mean.strip().lower() for mean in read_mean]

			updated_evec_file = os.path.join("data","updated_eigenvectors_{}".format(args.resolution),cell,\
				"{}_eigen_{}_chr{}.csv".format(cell,args.resolution,ch))
			f = open(updated_evec_file, "r")
			lines = f.readlines()
			eigenvectors = lines[1:]
			eigenvectors = [evec.split(',')[1].strip().lower() for evec in eigenvectors]

			evec_np = np.array([np.nan if line=="" else np.float(line) for line in eigenvectors])

			if len(cell_evecs)==0:
				cell_evecs = evec_np
			else:
				cell_evecs = np.concatenate((cell_evecs,evec_np))

			
			filter_means = []

			count = 0

			for idx, evec in enumerate(eigenvectors):
				if evec == "":
					continue
				else:
					filter_means.append(all_mean[idx])

			mean_evec["{}_chr{}".format(cell,ch)] = filter_means

		all_evec[cell] = cell_evecs

	cleaned_data = {}
	

	for key, item in all_data.items():
		cell = key.split("_")[0]
		chr_num = key.split("_")[1]
		curr_mean_evec = mean_evec[key]
		# print(key)
		cleaned_data[key] = clean_data_with_mean(item,None,curr_mean_evec)

	return cleaned_data


def prepare_data_cross_validation(args,cell_lines):
	
	# cell_lines = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	# cell_lines = ["IMR90", "NHEK"]
	# read all data, store with key cellName_chrX: all data
	all_data = {}
	train_data_all = {}
	test_data_all = {}

	for cell_line in cell_lines:
		cell_line_dir = os.path.join(args.data_dir,cell_line)
		for filename in os.listdir(cell_line_dir):
			print("***Reading data from {}...".format(filename))
			chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as csvfile:
				data_reader = csv.reader(csvfile, delimiter=',')
				lines = []
				for row in data_reader:
					row = [val if val else 'NA' for val in row]
					lines.append(row)
				all_data[chromosome] = lines

	# compute mean value for each histone modification, replace NA with mean

	data_info = {0:{"sum":0, "count":0},\
				1:{"sum":0, "count":0},\
				2:{"sum":0, "count":0},\
				3:{"sum":0, "count":0},\
				4:{"sum":0, "count":0},\
				5:{"sum":0, "count":0}
				}
	empty_point = defaultdict(list)

	for chrom, data in all_data.items():
		for line_id, line in enumerate(data):
			for hm_idx, value in enumerate(line[3:]):
				if value == 'NA':
					empty_point[hm_idx].append((chrom,line_id))
				else:
					sum_value = data_info[hm_idx]["sum"] + float(value)
					data_count = data_info[hm_idx]["count"] + 1
					data_info[hm_idx]["sum"] = sum_value
					data_info[hm_idx]["count"] = data_count

	mean_values = {}
	print(data_info)
	for hm_idx, info in data_info.items():
		# print(hm_idx,info)
		mean_values[hm_idx] = info["sum"] / info["count"]

	# print(mean_values)

	for hm_idx, point_list in empty_point.items():
		mean_value = mean_values[hm_idx]
		for (chrom, line_id) in point_list:
			all_data[chrom][line_id][hm_idx+3] = mean_value

	print("***Finish Imputing***")

	#clean data
	cleaned_data = {}
	for key, item in all_data.items():
		cell = key.split("_")[0]
		chr_num = key.split("_")[1]
		if args.use_corre:
			corre = correlations[cell][chr_num]
		else:
			corre = None
		cleaned_data[key] = clean_data(item,corre)

	#divide data into train, validate, testing	
	train_chrom = {}
	validate_chrom = {}
	test_chrom = {}

	

	for chrom, data in cleaned_data.items():
		# print(chrom)
		cell = chrom.split("_")[0]
		chr_num = chrom.split("_")[1][3:]

		if chr_num == "X":
			continue
		chr_num	 = int(chr_num)	
		if cell == args.cell:
			test_chrom[chrom] = data
		else:
			train_chrom[chrom] = data

	print("***Finish Cleaning***")
	
	
	# print(test_chrom.keys())
	#combine data
	training_data = combine_data(train_chrom)
	testing_data = combine_data(test_chrom)
	print("***Finish Combining***")

	#save test loader
	test_dataset = abDataset(testing_data)
	test_loader = DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False)
	torch.save(test_loader,os.path.join(args.data_save_dir,"test.pth"))
	


	# train_dataset = abDataset(training_data)
	# valid_dataset = abDataset(validation_data)
	# train_loader = DataLoader(train_dataset,batch_size=args.train_batch_size, shuffle=True)
	# valid_loader = DataLoader(valid_dataset,batch_size=args.valid_batch_size, shuffle=False)
	


	#spit data for cross-validation
	kfold =KFold(n_splits=args.num_fold)
	for fold, (train_index, valid_index) in enumerate(kfold.split(training_data)):
		print("TRAIN:", train_index, "TEST:", valid_index)
		train_fold = [training_data[i] for i in train_index]
		valid_fold = [training_data[i] for i in valid_index]

		train_dataset = abDataset(train_fold)
		valid_dataset = abDataset(valid_fold)

		train_loader = DataLoader(train_dataset,batch_size=args.train_batch_size, shuffle=True)
		valid_loader = DataLoader(valid_dataset,batch_size=args.valid_batch_size, shuffle=False)

		fold_save_path = os.path.join(args.data_save_dir,str(fold))
		torch.save(train_loader,os.path.join(fold_save_path,"train.pth"))
		torch.save(valid_loader,os.path.join(fold_save_path,"valid.pth"))

	return testing_data



def prepare_data_cell_lines_single(args,data_dir, target_cell, use_corre = False):
	'''
	Version 6

	Use chr 1-16 of all cell lines for training
	Use chr 17-18 of IMR90 for validation

	use mean value to fill up missing values

	'''
	cell_lines = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	#cell_lines = [target_cell]
	
	# read all data, store with key cellName_chrX: all data
	all_data = {}
	train_data_all = {}
	test_data_all = {}

	correlations = {}

	corre_dpath = os.path.join("data","6_cell_{}_correlations.json".format(args.resolution))

	if args.use_corre:
		if os.path.isfile(corre_dpath):
			print("reading correlation file: {}...".format(corre_dpath))
			with open(corre_dpath, 'r') as j:
				correlations = json.loads(j.read())
		else:
			print("no correlation file found")
			exit(0)
			#for cell in cell_lines:
			#	correlations[cell] = calculate_correlation(cell)
			#with open(corre_dpath, 'w') as outfile:
			#	json.dump(correlations, outfile)

	for cell_line in cell_lines:
		cell_line_dir = os.path.join(args.data_dir, cell_line)
		for filename in os.listdir(cell_line_dir):
			print("***Reading data from {}...".format(filename))
			chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as csvfile:
				data_reader = csv.reader(csvfile, delimiter=',')
				lines = []
				for row in data_reader:
					row = [val if val else 'NA' for val in row]
					lines.append(row)
				all_data[chromosome] = lines

	# compute mean value for each histone modification, replace NA with mean

	data_info = {0:{"sum":0, "count":0},\
				1:{"sum":0, "count":0},\
				2:{"sum":0, "count":0},\
				3:{"sum":0, "count":0},\
				4:{"sum":0, "count":0},\
				5:{"sum":0, "count":0}
				}
	empty_point = defaultdict(list)

	for chrom, data in all_data.items():
		for line_id, line in enumerate(data):
			for hm_idx, value in enumerate(line[3:]):
				# print(hm_idx)
				# print(value)
				if value == 'NA':
					empty_point[hm_idx].append((chrom,line_id))
				else:
					sum_value = data_info[hm_idx]["sum"] + float(value)
					data_count = data_info[hm_idx]["count"] + 1
					data_info[hm_idx]["sum"] = sum_value
					data_info[hm_idx]["count"] = data_count

	mean_values = {}
	print(data_info)
	for hm_idx, info in data_info.items():
		# print(hm_idx,info)
		mean_values[hm_idx] = info["sum"] / info["count"]

	# print(mean_values)

	for hm_idx, point_list in empty_point.items():
		mean_value = mean_values[hm_idx]
		for (chrom, line_id) in point_list:
			all_data[chrom][line_id][hm_idx+3] = mean_value

	print("***Finish Imputing***")

	#clean data
	cleaned_data = {}
	for key, item in all_data.items():
		cell = key.split("_")[0]
		chr_num = key.split("_")[1]
		if use_corre:
			corre = correlations[cell][chr_num]
		else:
			corre = None
		cleaned_data[key] = clean_data(item,corre)

	#divide data into train, validate, testing	
	train_chrom = {}
	validate_chrom = {}
	test_chrom = {}

	print("***Finish Cleaning***")

	for chrom, data in cleaned_data.items():
		# print(chrom)
		cell = chrom.split("_")[0]
		chr_num = chrom.split("_")[1][3:]
		# print("target cell: {}".format(target_cell))
		# print(cell)
		# print(chr_num)

		if cell == args.cell:
			if chr_num == "X":
				continue

			chr_num	 = int(chr_num)	
			if chr_num >= 17 and chr_num < 19:
				# print("add valid data")
				validate_chrom[chrom] = data
			elif chr_num >= 19 and chr_num < 23:
				# print("add test data")
				test_chrom[chrom] = data
			elif chr_num < 17:
				# print("add train data")
				train_chrom[chrom] = data
	print("***Finish Combining***")
	# print(test_chrom.keys())
	#combine data
	training_data = combine_data(train_chrom)
	validation_data = combine_data(validate_chrom)
	testing_data = combine_data(test_chrom)

	print(training_data[0])


	return training_data, validation_data, testing_data


def combine_plot_curve(split_list,curve_names):
	data_dpath = os.path.join("data","exp_data","curve_info.json")
	data = {}
	if os.path.isfile(data_dpath):
		with open(data_dpath, 'r') as j:
			data = json.loads(j.read())

	now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	exp_dpath = os.path.join("data","exp_data","exp_{}".format(now))
	if not os.path.exists(exp_dpath):
		os.makedirs(exp_dpath)

	experiment.log_text(now)

	#plot auroc curve
	pyplot.title("AUROC")
	img_path = os.path.join(exp_dpath,"auroc.png")
	for idx, name in enumerate(split_list):
		fpr, tpr =  data[name]["auroc"]["fpr"], data[name]["auroc"]["tpr"]
		pyplot.plot(fpr, tpr, linestyle='--', label= "{}: {:.3f}".format(curve_names[idx],data[name]["auroc_score"]))
		# axis labels
	
	pyplot.xlabel('False Positive Rate')
	pyplot.ylabel('True Positive Rate')
	# show the legend
	pyplot.legend()
	pyplot.savefig(img_path)
	pyplot.clf()
	experiment.log_image(img_path)

	#plot auprc curve
	pyplot.title("AUPRC")
	img_path = os.path.join(exp_dpath,"auprc.png")
	for idx, name in enumerate(split_list):
		recall, precision =  data[name]["auprc"]["recall"], data[name]["auprc"]["precision"]
		pyplot.plot(recall, precision, marker='.', label="{}: {:.3f}".format(curve_names[idx],data[name]["auprc_score"]))
		# axis labels
	
	pyplot.xlabel('False Positive Rate')
	pyplot.ylabel('True Positive Rate')
	# show the legend
	pyplot.legend()
	pyplot.savefig(img_path)
	pyplot.clf()
	experiment.log_image(img_path)



def load_validation():
	cell_line_dir = "data/validation"
	validation = ["IMR90_chr17.data.csv","IMR90_chr18.data.csv"]
	all_data = {}
	num_lines = 0
	for filename in validation:
		print("***Reading data from {}...".format(filename))
		chromosome = filename.split(".")[0]
		filepath = os.path.join(cell_line_dir,filename)
		with open(filepath, newline='') as csvfile:
			data_reader = csv.reader(csvfile, delimiter=',')
			lines = []
			for row in data_reader:
				row = [val if val else 'NA' for val in row]
				lines.append(row)
				num_lines +=1
			all_data[chromosome] = lines

	cleaned_data = {}
	print("lines of data: ",num_lines)
	for key, item in all_data.items():
		cleaned_data[key] = clean_data(item)
	print("len of cleaned data: ",len(cleaned_data))
	for chrom, data in cleaned_data.items():
		print("key {}, num of cleaned data: {}".format(chrom,len(data)))
	combined = combine_data(cleaned_data)
	print("len of combined data: ",len(combined))
	return combined

def plot_avg_evecs(experiment):
	predict_data_path = "data/avg_evecs_updated.txt"
	f = open(predict_data_path, "r")
	predicts = f.readlines()

	target_data_dir = "data/validation"

	validation = ["IMR90_eigen_100kb_chr17.csv","IMR90_eigen_100kb_chr18.csv"]

	all_lines = []
	for file in validation:
		file_path = os.path.join(target_data_dir,file)
		f = open(file_path, "r")
		lines = f.readlines()
		
		all_lines += lines[1:]

	targets = []
	for line in all_lines:
		target = line.strip().split(",")[1]
		try: 
			target = float(target)
			targets.append(target)
		except:
			targets.append("nan")

	print(len(targets))
	print(len(predicts))

	new_targets = []
	new_predicts = []


	for idx, (predict, target) in enumerate(zip(predicts,targets)):
		predict = predict.strip()
		if predict == "nan" or target == "nan":
			continue
		else:
			new_targets.append(float(1 if target > 0 else 0))
			new_predicts.append(float(predict))


	img_path = "mean_eigens_auroc"
	curve_name = "AUROC for avg eigenvectors"
	save_fig = True

	auroc,fpr,tpr = AUROC(experiment, new_targets, new_predicts, img_path, curve_name,save_fig)
	print(auroc)
	experiment.log_metric("AUROC",auroc)

	curve_name = "AUPRC for avg eigenvectors"
	img_path = "mean_eigens_auprc"
	auprc,recall, precision = AUPRC(experiment, new_targets, new_predicts, img_path, "AUPRC", save_fig)
	

	#save best auroc and best auprc data
	auroc_data, auprc_data = {"fpr":fpr.tolist(), "tpr":tpr.tolist()}, {"recall":recall.tolist(),"precision":precision.tolist()}
	

	save_dpath = os.path.join("data","exp_data")
	if not os.path.exists(save_dpath):
		os.makedirs(save_dpath)

	save_file = os.path.join(save_dpath,"curve_info_imr90+v4.json")
	save_data = {}
	if os.path.isfile(save_file):
		with open(save_file, 'r') as j:
			save_data = json.loads(j.read())

	save_data["mean"] = {"auroc_score":auroc,"auprc_score":auprc, "auprc":auprc_data,"auroc":auroc_data}

	with open(save_file, 'w') as outfile:
		json.dump(save_data, outfile)

	print(auprc)
	experiment.log_metric("AUPRC",auprc)

# experiment = Experiment(
#     api_key="pTKIV6STi3dM8kJmovb0xt0Ed",
#     project_name="a-b-prediction",
#     workspace="suchzheng2",
# )

# plot_avg_evecs(experiment)


def analyse_data(data_dir):
	'''
	Version 3 

	Use chr 1-16 of all cell lines for training
	Use chr 17-18 of IMR90 for validation

	use mean value to fill up missing values

	'''
	# data_dir = os.path.join("data","6_cell_lines")
	cell_lines = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	
	# read all data, store with key cellName_chrX: all data
	all_data = {}
	train_data_all = {}
	test_data_all = {}

	for cell_line in cell_lines:
		cell_line_dir = os.path.join(data_dir,cell_line)
		for filename in os.listdir(cell_line_dir):
			print("***Reading data from {}...".format(filename))
			chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as csvfile:
				data_reader = csv.reader(csvfile, delimiter=',')
				lines = []
				for row in data_reader:
					row = [val if val else 'NA' for val in row]
					lines.append(row)
				all_data[chromosome] = lines

	# compute mean value for each histone modification, replace NA with mean

	data_info = {0:{"sum":0, "count":0},\
				1:{"sum":0, "count":0},\
				2:{"sum":0, "count":0},\
				3:{"sum":0, "count":0},\
				4:{"sum":0, "count":0},\
				5:{"sum":0, "count":0}
				}
	empty_point = defaultdict(list)

	for chrom, data in all_data.items():
		for line_id, line in enumerate(data):
			for hm_idx, value in enumerate(line[3:]):
				# print(hm_idx)
				# print(value)
				if value == 'NA':
					empty_point[hm_idx].append((chrom,line_id))
				else:
					sum_value = data_info[hm_idx]["sum"] + float(value)
					data_count = data_info[hm_idx]["count"] + 1
					data_info[hm_idx]["sum"] = sum_value
					data_info[hm_idx]["count"] = data_count

	mean_values = {}
	print(data_info)
	for hm_idx, info in data_info.items():
		# print(hm_idx,info)
		mean_values[hm_idx] = info["sum"] / info["count"]

	# print(mean_values)

	for hm_idx, point_list in empty_point.items():
		mean_value = mean_values[hm_idx]
		for (chrom, line_id) in point_list:
			all_data[chrom][line_id][hm_idx+3] = mean_value

	#clean data
	cleaned_data = {}
	for key, item in all_data.items():
		cleaned_data[key] = clean_data(item)

	#divide data into train, validate, testing	
	train_chrom = {}
	validate_chrom = {}
	test_chrom = {}


	all_data_info = {}
	for chrom,data in cleaned_data.items():
		cell = chrom.split("_")[0]
		chr_num = chrom.split("_")[1][3:]

		pos_count = 0
		neg_count = 0

		if cell not in all_data_info.keys():
			all_data_info[cell] = {}

		for item in data:
			if item["cla_label"] == 1:
				pos_count += 1
			else:
				neg_count += 1
		all_data_info[cell][chr_num] = {"positive": pos_count, "negative": neg_count}

	save_path = os.path.join("data","6_cell_line_data_info.txt")

	save_file = open(save_path,"w+")

	for cell in cell_lines:
		total_pos = 0
		total_neg = 0
		print("========= {} =========".format(cell))
		save_file.write("========= {} =========\n".format(cell))
		for chr_num in range(1,23):
			data = all_data_info[cell][str(chr_num)]
			total_pos += data["positive"]
			total_neg += data["negative"]
			print("{}-{}:\tpositive: {},\tnegative: {}".format(cell,chr_num,data["positive"],data["negative"]))
			save_file.write("{}-{}:\tpositive: {},\tnegative: {}\n".format(cell,chr_num,data["positive"],data["negative"]))

		print("{} Summary:\tpostive:{}\tnegative:{}\n\n".format(cell,total_pos,total_neg))
		save_file.write("{} Summary:\tpostive:{}\tnegative:{}\n".format(cell,total_pos,total_neg))

def analyze_eigenvector(data_dir):

	# data_dir = os.path.join("data","6_cell_lines")
	# cell_lines = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	cell_lines = ["HMEC", "IMR90"]
	# read all data, store with key cellName_chrX: all data
	all_data = {}
	train_data_all = {}
	test_data_all = {}

	all_data_info = {}

	for cell in cell_lines:
		cell_line_dir = os.path.join(data_dir,cell,"100kb","reprocessed_evecs")
		# cell_line_dir = os.path.join(data_dir,cell,"100kb")
		for idx in range(1,23):
			filename = "{}_eigen_100kb_chr{}.txt".format(cell,idx)
		
		# for filename in os.listdir(cell_line_dir):
			
			# chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as file:
				lines =  file.readlines()

			if cell not in all_data_info.keys():
				all_data_info[cell] = {}

			pos_count = 0
			neg_count = 0
			for item in lines:
				item = item.strip().lower()
				if item == "nan":
					continue
				if float(item) > 0:
					pos_count += 1
				else:
					neg_count += 1
			all_data_info[cell][str(idx)] = {"positive": pos_count, "negative": neg_count}

		
	save_path = os.path.join("data","reprocessed_6_cell_line_data_info.txt")

	save_file = open(save_path,"w+")

	for cell in cell_lines:
		total_pos = 0
		total_neg = 0
		print("========= {} =========".format(cell))
		save_file.write("========= {} =========\n".format(cell))
		for chr_num in range(1,23):
			data = all_data_info[cell][str(chr_num)]
			total_pos += data["positive"]
			total_neg += data["negative"]
			print("{}-{}:\tpositive: {},\tnegative: {}".format(cell,chr_num,data["positive"],data["negative"]))
			save_file.write("{}-{}:\tpositive: {},\tnegative: {}\n".format(cell,chr_num,data["positive"],data["negative"]))

		print("{} Summary:\tpostive:{}\tnegative:{}\n\n".format(cell,total_pos,total_neg))
		save_file.write("{} Summary:\tpostive:{}\tnegative:{}\n".format(cell,total_pos,total_neg))


def analyze_eigenvector_updated(data_dir):

	# data_dir = os.path.join("data","6_cell_lines")
	cell_lines = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	# cell_lines = ["HMEC", "IMR90"]
	# read all data, store with key cellName_chrX: all data
	all_data = {}
	train_data_all = {}
	test_data_all = {}

	all_data_info = {}

	for cell in cell_lines:
		cell_line_dir = os.path.join(data_dir,cell,"100kb","updated")
		
		# cell_line_dir = os.path.join(data_dir,cell,"100kb","reprocessed_evecs")
		# cell_line_dir = os.path.join(data_dir,cell,"100kb")
		for idx in range(1,23):
			filename = "{}_eigen_100kb_chr{}.csv".format(cell,idx)
		
		# for filename in os.listdir(cell_line_dir):
			
			# chromosome = filename.split(".")[0]
			filepath = os.path.join(cell_line_dir,filename)
			with open(filepath, newline='') as file:
				lines =  file.readlines()

			if cell not in all_data_info.keys():
				all_data_info[cell] = {}

			pos_count = 0
			neg_count = 0
			for item in lines[1:]:
				item = item.split(",")[1]
				item = item.strip().lower()
				# print(item)
				if item == "nan":
					continue
				try:
					item = float(item)
				except:
					continue
				if item > 0:
					pos_count += 1
				else:
					neg_count += 1
			all_data_info[cell][str(idx)] = {"positive": pos_count, "negative": neg_count}

		
	save_path = os.path.join("data","updated_6_cell_line_data_info.txt")

	save_file = open(save_path,"w+")

	for cell in cell_lines:
		total_pos = 0
		total_neg = 0
		print("========= {} =========".format(cell))
		save_file.write("========= {} =========\n".format(cell))
		for chr_num in range(1,23):
			data = all_data_info[cell][str(chr_num)]
			total_pos += data["positive"]
			total_neg += data["negative"]
			print("{}-{}:\tpositive: {},\tnegative: {}".format(cell,chr_num,data["positive"],data["negative"]))
			save_file.write("{}-{}:\tpositive: {},\tnegative: {}\n".format(cell,chr_num,data["positive"],data["negative"]))

		print("{} Summary:\tpostive:{}\tnegative:{}\n\n".format(cell,total_pos,total_neg))
		save_file.write("{} Summary:\tpostive:{}\tnegative:{}\n".format(cell,total_pos,total_neg))

	
# analyse_data()

# analyze_eigenvector_updated(os.path.join("data","Compartments"))


# analyze_eigenvector(os.path.join("data","Compartments"))




	