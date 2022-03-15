"""
This file contains the code to load several models, test models and generate a combined curve
"""

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
from preprocess import *
from model import *
from utils import *
import json
import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)

def to_binary(val):
	if float(val) >= 0.5:
		return 1
	else:
		return 0

def test(model,test_loader,task,use_mask = False, mask = None):
	# print("===Testing...")
	model.eval()
	loss_fn = nn.CrossEntropyLoss()
	cur_loss = []
	cur_pred = []
	cur_labels = []
	cur_mean = []
	mean_evec = None
	for batch in tqdm(test_loader):
		# print(batch)
		inputs = batch["input"]
		inputs = inputs.to(device)
		if task == "cla":
			labels = batch["cla_labels"]
		elif task == "reg":
			labels = batch["reg_labels"]
		
		try:
			mean_evec = batch["mean_evec"]
			mean_evec = mean_evec.to(device)
		except:
			mean_evec = None
		labels = labels.to(device)
		if use_mask:
			mask = mask.to(device)

		y_pred = model.forward(inputs,mean_evec,use_mask,mask)
		loss = loss_fn(y_pred, labels)
		loss = loss.cpu().detach().numpy()
		cur_loss.append(loss)
		
		cur_pred += y_pred[:,-1].detach().tolist()
		cur_mean += mean_evec.detach().tolist()
		cur_labels += labels.detach().tolist()
	


	curve_name = "AUROC"
	auroc, fpr, tpr = AUROC(experiment,cur_labels,cur_pred, None, curve_name,save_fig = False)
	
	curve_name = "AUPRC"
	auprc, recall, precision = AUPRC(experiment,cur_labels,cur_pred, None, curve_name,save_fig = False)

	#calculate accuracy
	pred_binary = [to_binary(val) for val in cur_pred]
	accuracy = accuracy_score(cur_labels, pred_binary)

	mean_auroc, fpr, tpr = AUROC(experiment,cur_labels,cur_mean, None, curve_name,save_fig = False)
	mean_auprc, recall, precision = AUPRC(experiment,cur_labels,cur_mean, None, curve_name,save_fig = False)
	print("model: auroc - {}, auprc - {}".format(auroc,auprc))
	print("mean: auroc - {}, auprc - {}".format(mean_auroc,mean_auprc))
	
	data = {}
	data["auroc"] = {}
	data["auroc"]["auroc_score"] = auroc
	data["auroc"]["fpr"] = fpr
	data["auroc"]["tpr"] = tpr

	data["auprc"] = {}
	data["auprc"]["auprc_score"] = auprc
	data["auprc"]["recall"] = recall
	data["auprc"]["precision"] = precision

	data["accuracy"] = accuracy
	
	return data


def combine_plot_curve(curve_config):

	data = curve_config["data"]
	exp_dpath = curve_config["exp_dpath"]

	pyplot.title("AUROC ({})".format(curve_config["figure_name"]))
	img_path = os.path.join(exp_dpath,"{}_auroc.png".format(curve_config["cell"]))

	for name in data.keys():
		# print(data[name])
		fpr, tpr =  data[name]["auroc"]["fpr"], data[name]["auroc"]["tpr"]
		pyplot.plot(fpr, tpr, linestyle='--', label= "{}: {:.3f}".format(name,data[name]["auroc"]["auroc_score"]))
		# axis labels

	pyplot.xlabel('False Positive Rate')
	pyplot.ylabel('True Positive Rate')
	# show the legend
	pyplot.legend()
	pyplot.savefig(img_path)
	# pyplot.show()
	pyplot.clf()
	experiment.log_image(img_path)

	#plot auprc curve
	pyplot.title("AUPRC ({})".format(curve_config["figure_name"]))
	img_path = os.path.join(exp_dpath,"{}_auprc.png".format(curve_config["cell"]))

	for name in data.keys():
		recall, precision =  data[name]["auprc"]["recall"], data[name]["auprc"]["precision"]
		pyplot.plot(recall, precision, marker='.', label="{}: {:.3f}".format(name,data[name]["auprc"]["auprc_score"]))
		# axis labels
	
	pyplot.xlabel('Recall')
	pyplot.ylabel('Precision')
	# show the legend
	pyplot.legend()
	pyplot.savefig(img_path)
	pyplot.clf()
	experiment.log_image(img_path)


def generate_curve(models_info, exp_dpath):

	print(models_info)
	print(exp_dpath)


	curve_info = {}
	test_results = {}
	curve_info["exp_dpath"] = exp_dpath
	curve_info["figure_name"] = models_info[2]
	curve_info["cell"] = models_info[0]

	do_test = False

	if models_info[1] == "test":
		do_test = True
	elif models_info[0] == "valid":
		do_test == False


	for curr_model in models_info[3:]:
		if curr_model["type"] == "model":
			results = load_model_and_run(curr_model, do_test)
			label = curr_model["label"]
			test_results[label] = results
		elif curr_model["type"] == "mean_baseline":
			results = load_mean_and_compare(curr_model,do_test)
			label = curr_model["label"]
			test_results[label] = results
		elif curr_model["type"] == "mean_by_region":
			results = load_mean_by_region(curr_model,do_test)
			label = curr_model["label"]
			test_results[label] = results
		elif curr_model["type"] == "model_by_region":
			for idx in range(curr_model["number_of_region"]):
				results = load_model_and_run_by_region(curr_model,idx,do_test)
				label = "region_{}".format(idx)
				test_results[label] = results
		elif curr_model["type"] == "mean_by_multiple_region":
			for idx in range(curr_model["number_of_region"]):
				print("region {}".format(idx))
				curr_model["predict_dir"] = os.path.join(curr_model["data_dir"],\
					"region_{}".format(idx),\
					"{}_{}_region{}_predict.txt".format(curr_model["target"],curr_model["resolution"],idx))
				curr_model["target_dir"] = os.path.join(curr_model["data_dir"],\
					"region_{}".format(idx),\
					"{}_{}_region{}_target.txt".format(curr_model["target"],curr_model["resolution"],idx))
				results = load_mean_by_region(curr_model,do_test)
				label = "region_{}_mean".format(idx)
				test_results[label] = results
		elif curr_model["type"] == "mean_baseline_by_chr":
			results = load_mean_and_compare_by_chr(curr_model,do_test)
			label = curr_model["label"]
			test_results[label] = results
		elif curr_model["type"] == "perturbation_test":
			#save results to a .csv file
			save_file = os.path.join(exp_dpath,"{}_perturbation.csv".format(curr_model["target"]))
			with open(save_file,"w", newline='') as csvfile:
				writer = csv.writer(csvfile)
				writer.writerow(["cell","H3K4me1", "H3K4me3", "H3K27ac", "H3K27me3","H3K9me3","H3K36me3","AUROC","Accuracy"])
				writer.writerow([curr_model["target"],0,0,0,0,0,0,test_results["GRU"]["auroc"]["auroc_score"],test_results["GRU"]["accuracy"]])
				writer.writerow([curr_model["target"],1,1,1,1,1,1,test_results["Mean"]["auroc"]["auroc_score"],test_results["Mean"]["accuracy"]])

				results = load_model_and_perform_pertutbation(curr_model,do_test)

				for result in results:
					label = "remove: " + result["label"]
					test_results[label] = result

					hm_list = label.split(",")
					print(hm_list)
					print_line = [curr_model["target"]]
					for i in range(0,6):
						if i in result["selection"]:
							print_line.append(1)
						else:
							print_line.append(0)
					print_line.append(result["auroc"]["auroc_score"])
					print_line.append(result["accuracy"])
					writer.writerow(print_line)



	curve_info["data"] = test_results

	combine_plot_curve(curve_info)

	# print_line = ''
	# results_values=list(test_results.values())
	# for data in results_values:
	# 	print_line += "{:.3f},".format(data["auroc"]["auroc_score"])

	# print("auroc: {}".format(print_line))

	# for data in results_values:
	# 	print_line += "{:.3f},".format(data["accuracy"])

	# print("accuracy: {}".format(print_line))

def generate_mask(batch,d1,d2,mask_index):
	mask = torch.ones(batch,d1,d2)
	for idx in mask_index:
		mask[0,:,idx] = 0
	return mask

def select_hm(total_hm,hm_num):
	hm = list(range(total_hm))
	return list(itertools.combinations(hm,hm_num))


def load_model_and_perform_pertutbation(curr_model,do_test):
	agg_results = []
	
	if do_test:
		print(curr_model["dataloader_dir"])
		test_loader = torch.load(os.path.join(curr_model["dataloader_dir"]))
	else:
		valid_loader = torch.load(os.path.join(curr_model["dataloader_dir"]))

	load_path = os.path.join("data","exp_data","cla","6_cell_input_updated_{}"\
		.format(curr_model["resolution"]),\
				curr_model["model_type"],\
				curr_model["run"])
	print(load_path)
	model_param_path = os.path.join(load_path,"model_param.json")

	model_param = {}
	print()
	if os.path.isfile(model_param_path):
		with open(model_param_path, 'r') as j:
			model_param = json.loads(j.read())
	print(model_param)

	#load model
	if curr_model["model_type"] == "lstm":
		model = lstm(100,model_param).to(device)
	elif curr_model["model_type"] == "gru":
		model = gru(100,model_param).to(device)
	elif curr_model["model_type"][:3] == "cnn":
		n = model_type.split("_")[1]
		model = cnnNlayer(100,model_param, "cla").to(device)

	model.load_state_dict(torch.load(os.path.join(load_path,'model.pt')))



	histone_modifications = ["H3K4me1", "H3K4me3", "H3K27ac", "H3K27me3","H3K9me3","H3K36me3"]
	number_of_hm = [1,2,3,4,5]
	# number_of_hm = [1]
	for hm_num in number_of_hm:
		combinations = select_hm(6,hm_num)
		for comb in combinations:
			#The combination is the histone modification we want to remove
			mask = generate_mask(1,100,6,comb)
			print("cell: {}, mask: {}".format(curr_model["target"],comb))

			result = test(model,test_loader,"cla",True,mask)
			label = ""
			for i in comb:
				label = label+histone_modifications[i]+","

			result["label"] = label[:-1]
			result["selection"] = comb
			print(result["auroc"]["auroc_score"])
			agg_results.append(result)



	# for idx, hm in enumerate(histone_modifications):
	# 	mask = generate_mask(1,100,6,idx)
	# 	# print(mask)
	# 	print("cell: {}, remove: {}".format(curr_model["target"],hm))

	# 	result = test(model,test_loader,"cla",True,mask)
	# 	result["label"] = "remove {}".format(hm)
	# 	print(result["auroc"]["auroc_score"])
	# 	agg_results.append(result)

	return agg_results



def load_model_and_run_by_region(curr_model,region_id,do_test):
	print(do_test)

	if do_test:
		print(curr_model["dataloader_dir"])
		test_loader = torch.load(os.path.join(curr_model["dataloader_dir"],\
			"region_{}".format(region_id),
			"{}_{}_region{}_test.pth").format(curr_model["target"],curr_model["resolution"],region_id))


	load_path = os.path.join("data","exp_data","cla","6_cell_input_updated_{}"\
		.format(curr_model["resolution"]),\
				curr_model["model_type"],\
				curr_model["run"])
	print(load_path)
	model_param_path = os.path.join(load_path,"model_param.json")

	model_param = {}
	print()
	if os.path.isfile(model_param_path):
		with open(model_param_path, 'r') as j:
			model_param = json.loads(j.read())
	print(model_param)

	#load model
	if curr_model["model_type"] == "lstm":
		model = lstm(100,model_param).to(device)
	elif curr_model["model_type"] == "gru":
		model = gru(100,model_param).to(device)
	elif curr_model["model_type"][:3] == "cnn":
		n = model_type.split("_")[1]
		model = cnnNlayer(100,model_param, "cla").to(device)

	model.load_state_dict(torch.load(os.path.join(load_path,'model.pt')))
	
	if do_test ==True:
		print("Do test")
		results = test(model,test_loader,"cla")
	else:
		results = test(model,valid_loader,"cla")

	print(results["auroc"]["auroc_score"])

	return results
def load_mean_by_region(curr_model,do_test):
	pred_f = open(curr_model["predict_dir"], "r")
	predicts = pred_f.readlines()
	
	tar_f = open(curr_model["target_dir"], "r")
	targets = tar_f.readlines()

	predicts = [line.strip() for line in predicts]
	targets = [line.strip() for line in targets]

	new_predicts = []
	new_targets = []

	for idx in range(len(predicts)):
		if predicts[idx] != "nan" and targets[idx] != "":
			new_predicts.append(float(predicts[idx]))
			new_targets.append(float(targets[idx]))



	img_path = "mean_eigens_auroc"
	curve_name = "AUROC for avg eigenvectors"
	save_fig = True
	# print(new_targets)
	# print(new_predicts)
	auroc,fpr,tpr = AUROC(experiment, new_targets, new_predicts, img_path, curve_name,save_fig)
	print(auroc)
	experiment.log_metric("AUROC",auroc)

	curve_name = "AUPRC for avg eigenvectors"
	img_path = "mean_eigens_auprc"
	auprc,recall, precision = AUPRC(experiment, new_targets, new_predicts, img_path, "AUPRC", save_fig)

	#calculate accuracy
	pred_binary = [to_binary(val) for val in new_predicts]
	accuracy = accuracy_score(new_targets, pred_binary)


	data = {}
	data["auroc"] = {}
	data["auroc"]["auroc_score"] = auroc
	data["auroc"]["fpr"] = fpr
	data["auroc"]["tpr"] = tpr

	data["auprc"] = {}
	data["auprc"]["auprc_score"] = auprc
	data["auprc"]["recall"] = recall
	data["auprc"]["precision"] = precision

	data["accuracy"] = accuracy


	return data

def load_mean_and_compare(curr_model,do_test = False):
	f = open(curr_model["data_dir"], "r")
	predicts = f.readlines()
	all_lines = []
	target_data_dir = os.path.join("data","updated_eigenvectors_{}".format(curr_model["resolution"]),\
		curr_model["target"])
	if curr_model["split"] == "5_cells":
		if do_test:
			chr_range = (19,23)
		else:
			chr_range = (17,19)
	elif curr_model["split"] =="cross_validation":
		if do_test:
			chr_range = (1,23)

	for idx in range(chr_range[0],chr_range[1]):
		file_path = os.path.join(target_data_dir,"{}_eigen_{}_chr{}.csv".format(curr_model["target"],curr_model["resolution"],idx))
		f = open(file_path, "r")
		lines = f.readlines()
		all_lines += lines[1:]

	targets = all_lines

	new_targets = []
	new_predicts = []

	nan_predict = []
	nan_target = []

	for idx, (predict, target) in enumerate(zip(predicts,targets)):
		
		predict = predict.strip().lower()
		target = target.split(',')[1].strip().lower()

		if predict == "nan":
			nan_predict.append(idx)
		if target == "":
			#print("target is nan")
			nan_target.append(idx)
		if predict != "nan" and target != "":
			new_targets.append(float(1 if float(target) > 0 else 0))
			new_predicts.append(float(predict))
	#print("nan_predict: {}".format(nan_predict))
	#print("nan_target: {}".format(nan_target))


	img_path = "mean_eigens_auroc"
	curve_name = "AUROC for avg eigenvectors"
	save_fig = True
	# print(new_targets)
	# print(new_predicts)
	auroc,fpr,tpr = AUROC(experiment, new_targets, new_predicts, img_path, curve_name,save_fig)
	print(auroc)
	experiment.log_metric("AUROC",auroc)

	curve_name = "AUPRC for avg eigenvectors"
	img_path = "mean_eigens_auprc"
	auprc,recall, precision = AUPRC(experiment, new_targets, new_predicts, img_path, "AUPRC", save_fig)

	#calculate accuracy
	pred_binary = [to_binary(val) for val in new_predicts]
	accuracy = accuracy_score(new_targets, pred_binary)





	data = {}
	data["auroc"] = {}
	data["auroc"]["auroc_score"] = auroc
	data["auroc"]["fpr"] = fpr
	data["auroc"]["tpr"] = tpr

	data["auprc"] = {}
	data["auprc"]["auprc_score"] = auprc
	data["auprc"]["recall"] = recall
	data["auprc"]["precision"] = precision

	data["accuracy"] = accuracy

	return data

def load_mean_and_compare_by_chr(curr_model,do_test = False):
	f = open(curr_model["data_dir"], "r")
	predicts = f.readlines()
	all_lines = []
	target_data_dir = os.path.join("data","updated_eigenvectors_{}".format(curr_model["resolution"]),\
		curr_model["target"])

	# target_data_dir = os.path.join("data","updated_eigenvectors_{}".format(curr_model["resolution"]),\
	# 	curr_model["target"],"{}_eigen_{}_chr{}".format(\
	# 		curr_model["target"]),
	# 		curr_model["resolution"],
	# 		curr_model["chr"])
	
	chrom = int(curr_model["chr"])
	for idx in range(chrom,chrom+1):
		file_path = os.path.join(target_data_dir,"{}_eigen_{}_chr{}.csv".format(curr_model["target"],curr_model["resolution"],idx))
		f = open(file_path, "r")
		lines = f.readlines()
		all_lines += lines[1:]

	targets = all_lines

	new_targets = []
	new_predicts = []

	nan_predict = []
	nan_target = []

	for idx, (predict, target) in enumerate(zip(predicts,targets)):
		
		predict = predict.strip().lower()
		target = target.split(',')[1].strip().lower()

		if predict == "nan":
			nan_predict.append(idx)
		if target == "":
			#print("target is nan")
			nan_target.append(idx)
		if predict != "nan" and target != "":
			new_targets.append(float(1 if float(target) > 0 else 0))
			new_predicts.append(float(predict))
	#print("nan_predict: {}".format(nan_predict))
	#print("nan_target: {}".format(nan_target))


	img_path = "mean_eigens_auroc"
	curve_name = "AUROC for avg eigenvectors"
	save_fig = True
	# print(new_targets)
	# print(new_predicts)
	auroc,fpr,tpr = AUROC(experiment, new_targets, new_predicts, img_path, curve_name,save_fig)
	print(auroc)
	experiment.log_metric("AUROC",auroc)

	curve_name = "AUPRC for avg eigenvectors"
	img_path = "mean_eigens_auprc"
	auprc,recall, precision = AUPRC(experiment, new_targets, new_predicts, img_path, "AUPRC", save_fig)

	#calculate accuracy
	pred_binary = [to_binary(val) for val in new_predicts]
	accuracy = accuracy_score(new_targets, pred_binary)





	data = {}
	data["auroc"] = {}
	data["auroc"]["auroc_score"] = auroc
	data["auroc"]["fpr"] = fpr
	data["auroc"]["tpr"] = tpr

	data["auprc"] = {}
	data["auprc"]["auprc_score"] = auprc
	data["auprc"]["recall"] = recall
	data["auprc"]["precision"] = precision

	data["accuracy"] = accuracy

	return data

def load_model_and_run(curr_model,do_test):

	print(do_test)

	if do_test:
		print(curr_model["dataloader_dir"])
		test_loader = torch.load(os.path.join(curr_model["dataloader_dir"]))
	else:
		valid_loader = torch.load(os.path.join(curr_model["dataloader_dir"]))

	load_path = os.path.join("data","exp_data","cla","6_cell_input_updated_{}"\
		.format(curr_model["resolution"]),\
				curr_model["model_type"],\
				curr_model["run"])
	print(load_path)
	model_param_path = os.path.join(load_path,"model_param.json")

	model_param = {}
	print()
	if os.path.isfile(model_param_path):
		with open(model_param_path, 'r') as j:
			model_param = json.loads(j.read())
	print(model_param)

	#load model
	if curr_model["model_type"] == "lstm":
		model = lstm(100,model_param).to(device)
	elif curr_model["model_type"] == "gru":
		model = gru(100,model_param).to(device)
	elif curr_model["model_type"][:3] == "cnn":
		n = model_type.split("_")[1]
		model = cnnNlayer(100,model_param, "cla").to(device)

	model.load_state_dict(torch.load(os.path.join(load_path,'model.pt')))
	
	if do_test ==True:
		print("Do test")
		results = test(model,test_loader,"cla")
	else:
		results = test(model,valid_loader,"cla")

	print(results["auroc"]["auroc_score"])

	return results

def plot_curve(data,exp_dpath):

	pyplot.title("AUROC")
	img_path = os.path.join(exp_dpath,"auroc.png")

	for name in data.keys():
		# print(data[name])
		fpr, tpr =  data[name]["auroc"]["fpr"], data[name]["auroc"]["tpr"]
		pyplot.plot(fpr, tpr, linestyle='--', label= "{}: {:.3f}".format(name,data[name]["auroc"]["auroc_score"]))
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
	for name in data.keys():
		recall, precision =  data[name]["auprc"]["recall"], data[name]["auprc"]["precision"]
		pyplot.plot(recall, precision, marker='.', label="{}: {:.3f}".format(name,data[name]["auprc"]["auprc_score"]))
		# axis labels
	
	pyplot.xlabel('Recall')
	pyplot.ylabel('Precision')
	# show the legend
	pyplot.legend()
	pyplot.savefig(img_path)
	pyplot.clf()
	experiment.log_image(img_path)

def main():

	now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	exp_dpath = os.path.join("data","exp_data","combined_curve","exp_{}".format(now))
	if not os.path.exists(exp_dpath):
		os.makedirs(exp_dpath)

	print(exp_dpath)

	experiment.log_text(now)




	# plot_id = [0,1,2,3,4]
	# plot_id = [6,7,8,9]
	# plot_id = [10,11,12,13,14,15] #
	# plot_id = [16,17,18,19,20,21]
	# plot_id = [22,23,24,25,26,27]
	# plot_id = [28,29,30,31,32],
	# plot_id = [33,34,35,36,37,38]
	# plot_id = [39,40,41,42,43]
	# plot_id = [47,48,49]
	# plot_id = [52,53,54,55,56]
	# plot_id = [57]

	# models_info = all_info[plot_id]

	# for idx in plot_id:
	# 	generate_curve(all_info[idx], exp_dpath)

	curve_config_dp = os.path.join("./data","curve_config","100kb_cell_gru_mean_test_strong.json")
	# curve_config_dp = os.path.join("./data","curve_config","100kb_gru_mean_pert.json")
	# curve_config_dp = os.path.join("./data","curve_config","test.json")

	with open(curve_config_dp) as json_file:
		data = json.load(json_file)

	for key,item in data.items():
		generate_curve(item, exp_dpath)

  

if __name__ == "__main__":
    main()




all_info = {
	#100kb no NHEK
		0:[
		"GM12878",
		"test",
		"GM12878 Testing, res: 100kb",
		{"type":"model","target": "GM12878", "model_type": "gru",\
			"run":"run_2021-06-02-18-02-06", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/5_cells_no_NHEK/GM12878/5_fold",
			"label":"gru (all, no NHEK)"},
		{"type":"mean_baseline","target":"GM12878","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome_no_NHEK/GM12878_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		1:[
		"K562",
		"test",
		"K562 Testing, res: 100kb",
		{"type":"model","target": "K562", "model_type": "gru",\
			"run":"run_2021-06-03-10-16-28", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/5_cells_no_NHEK/K562/5_fold",
			"label":"gru (all, no NHEK)"},
		{"type":"mean_baseline","target":"K562","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome_no_NHEK/K562_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		2:[
		"HMEC",
		"test",
		"HMEC Testing, res: 100kb",
		{"type":"model","target": "HMEC", "model_type": "gru",\
			"run":"run_2021-06-03-01-50-46", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/5_cells_no_NHEK/HMEC/5_fold",
			"label":"gru (all, no NHEK)"},
		{"type":"mean_baseline","target":"HMEC","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome_no_NHEK/HMEC_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		3:[
		"HUVEC",
		"test",
		"HUVEC Testing, res: 100kb",
		{"type":"model","target": "HUVEC", "model_type": "gru",\
			"run":"run_2021-06-07-13-19-07", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/5_cells_no_NHEK/HUVEC/5_fold",
			"label":"gru (all, no NHEK)"},
		{"type":"mean_baseline","target":"HUVEC","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome_no_NHEK/HUVEC_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		4:[
		"IMR90",
		"test",
		"IMR90 Testing, res: 100kb",
		{"type":"model","target": "IMR90", "model_type": "gru",\
			"run":"run_2021-06-02-21-59-10", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/5_cells_no_NHEK/IMR90/5_fold",
			"label":"gru (all, no NHEK)"},
		{"type":"mean_baseline","target":"IMR90","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome_no_NHEK/IMR90_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],
		5:[
		"GM12878",
		"test",
		"GM12878 Testing, res: 100kb",
		{"type":"model","target": "GM12878", "model_type": "gru",\
			"run":"run_2021-06-02-18-02-06", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/5_cells/IMR90-HMEC-K562-GM12878-HUVEC/GM12878/5_fold",
			"label":"gru (all, no NHEK)"},
		{"type":"mean_baseline","target":"GM12878","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome_no_NHEK/GM12878_test.txt",
			"label":"mean_evec_4cells","split":"cross_validation"},
		{"type":"model","target": "GM12878", "model_type": "gru",\
			"run":"run_2021-06-08-21-44-29", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/4_cells/IMR90-HMEC-K562-GM12878/GM12878/5_fold",
			"label":"gru (all, no NHEK,HUVEC)"},
		{"type":"mean_baseline","target":"GM12878","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome_no_NHEK_HUVEC/GM12878_test.txt",
			"label":"mean_evec_3cells","split":"cross_validation"}
		],
		#100kb no NHEK HUVEC
		
		6:[
		"GM12878",
		"test",
		"GM12878 Testing, res: 100kb",
		{"type":"model","target": "GM12878", "model_type": "gru",\
			"run":"run_2021-06-08-21-44-29", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/4_cells/IMR90-HMEC-K562-GM12878/GM12878/5_fold",
			"label":"gru (all, no NHEK,HUVEC)"},
		{"type":"mean_baseline","target":"GM12878","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome_no_NHEK_HUVEC/GM12878_test.txt",
			"label":"mean_evec_3cells","split":"cross_validation"}
		],
		7:[
		"IMR90",
		"test",
		"IMR90 Testing, res: 100kb",
		{"type":"model","target": "IMR90", "model_type": "gru",\
			"run":"run_2021-06-09-08-53-48", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/4_cells/IMR90-HMEC-K562-GM12878/IMR90/5_fold",
			"label":"gru (all, no NHEK, HUVEC)"},
		{"type":"mean_baseline","target":"IMR90","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome_no_NHEK_HUVEC/IMR90_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],
		8:[
		"HMEC",
		"test",
		"IMR90 Testing, res: 100kb",
		{"type":"model","target": "HMEC", "model_type": "gru",\
			"run":"run_2021-06-09-16-16-58", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/4_cells/IMR90-HMEC-K562-GM12878/HMEC/5_fold",
			"label":"gru (all, no NHEK, HUVEC)"},
		{"type":"mean_baseline","target":"HMEC","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome_no_NHEK_HUVEC/HMEC_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		9:[
		"K562",
		"test",
		"K562 Testing, res: 100kb",
		{"type":"model","target": "K562", "model_type": "gru",\
			"run":"run_2021-06-10-19-29-05", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/4_cells/IMR90-HMEC-K562-GM12878/K562/5_fold",
			"label":"gru (all, no NHEK, HUVEC)"},
		{"type":"mean_baseline","target":"K562","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome_no_NHEK_HUVEC/K562_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		#test by region, 100kb GRU
		10:[
		"GM12878",
		"test",
		"GM12878 Testing, res: 100kb",
		{"type":"model","target": "GM12878", "model_type": "gru",\
			"run":"run_2021-05-27-14-59-14", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/GM12878/5_fold/test.pth",
			"label":"All"},
		{"type":"model","target": "GM12878", "model_type": "gru",\
			"run":"run_2021-05-27-14-59-14", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/GM12878/GM12878_dynamic_test.pth",
			"label":"Dynamic region"},
		{"type":"model","target": "GM12878", "model_type": "gru",\
			"run":"run_2021-05-27-14-59-14", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/GM12878/GM12878_static_test.pth",
			"label":"Static region"},
		],

		11:[
		"K562",
		"test",
		"K562 Testing, res: 100kb",
		{"type":"model","target": "K562", "model_type": "gru",\
			"run":"run_2021-05-28-12-08-00", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/K562/5_fold/test.pth",
			"label":"All"},
		{"type":"model","target": "K562", "model_type": "gru",\
			"run":"run_2021-05-28-12-08-00", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/K562/K562_dynamic_test.pth",
			"label":"Dynamic region"},
		{"type":"model","target": "K562", "model_type": "gru",\
			"run":"run_2021-05-28-12-08-00", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/K562/K562_static_test.pth",
			"label":"Static region"},
		],

		12:[
		"NHEK",
		"test",
		"NHEK Testing, res: 100kb",
		{"type":"model","target": "NHEK", "model_type": "gru",\
			"run":"run_2021-05-28-04-10-01", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/NHEK/5_fold/test.pth",
			"label":"All"},
		{"type":"model","target": "NHEK", "model_type": "gru",\
			"run":"run_2021-05-28-04-10-01", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/NHEK/NHEK_dynamic_test.pth",
			"label":"Dynamic region"},
		{"type":"model","target": "NHEK", "model_type": "gru",\
			"run":"run_2021-05-28-04-10-01", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/NHEK/NHEK_static_test.pth",
			"label":"Static region"},
		],

		13:[
		"HMEC",
		"test",
		"HMEC Testing, res: 100kb",
		{"type":"model","target": "HMEC", "model_type": "gru",\
			"run":"run_2021-05-28-08-25-27", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/HMEC/5_fold/test.pth",
			"label":"All"},
		{"type":"model","target": "HMEC", "model_type": "gru",\
			"run":"run_2021-05-28-08-25-27", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/HMEC/HMEC_dynamic_test.pth",
			"label":"Dynamic region"},
		{"type":"model","target": "HMEC", "model_type": "gru",\
			"run":"run_2021-05-28-08-25-27", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/HMEC/HMEC_static_test.pth",
			"label":"Static region"},
		],

		14:[
		"HUVEC",
		"test",
		"HUVEC Testing, res: 100kb",
		{"type":"model","target": "HUVEC", "model_type": "gru",\
			"run":"run_2021-05-28-20-15-10", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/HUVEC/5_fold/test.pth",
			"label":"All"},
		{"type":"model","target": "HUVEC", "model_type": "gru",\
			"run":"run_2021-05-28-20-15-10", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/HUVEC/HUVEC_dynamic_test.pth",
			"label":"Dynamic region"},
		{"type":"model","target": "HUVEC", "model_type": "gru",\
			"run":"run_2021-05-28-20-15-10", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/HUVEC/HUVEC_static_test.pth",
			"label":"Static region"},
		],
		15:[
		"IMR90",
		"test",
		"IMR90 Testing, res: 100kb",
		{"type":"model","target": "IMR90", "model_type": "gru",\
			"run":"run_2021-05-27-22-38-47", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/IMR90/5_fold/test.pth",
			"label":"All"},
		{"type":"model","target": "IMR90", "model_type": "gru",\
			"run":"run_2021-05-27-22-38-47", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/IMR90/IMR90_dynamic_test.pth",
			"label":"Dynamic region"},
		{"type":"model","target": "IMR90", "model_type": "gru",\
			"run":"run_2021-05-27-22-38-47", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/IMR90/IMR90_static_test.pth",
			"label":"Static region"},
		],
		#test by region, 100kb GRU + mean

		16:[
		"GM12878",
		"test",
		"GM12878 Testing, res: 100kb",
		{"type":"model","target": "GM12878", "model_type": "gru",\
			"run":"run_2021-05-27-14-59-14", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/GM12878/5_fold/test.pth",
			"label":"All"},
		{"type":"model","target": "GM12878", "model_type": "gru",\
			"run":"run_2021-05-27-14-59-14", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/GM12878/GM12878_dynamic_test.pth",
			"label":"Dynamic region"},
		{"type":"model","target": "GM12878", "model_type": "gru",\
			"run":"run_2021-05-27-14-59-14", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/GM12878/GM12878_static_test.pth",
			"label":"Static region"},
		{"type":"mean_baseline","target":"GM12878","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/GM12878_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"mean_by_region","target":"GM12878","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/GM12878_dynamic_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/GM12878_dynamic_evec.txt",\
		"label":"Dynamic region (mean)"},
		{"type":"mean_by_region","target":"GM12878","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/GM12878_static_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/GM12878_static_evec.txt",\
		"label":"Static region (mean)"}
		],

		17:[
		"K562",
		"test",
		"K562 Testing, res: 100kb",
		{"type":"model","target": "K562", "model_type": "gru",\
			"run":"run_2021-05-28-12-08-00", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/K562/5_fold/test.pth",
			"label":"All"},
		{"type":"model","target": "K562", "model_type": "gru",\
			"run":"run_2021-05-28-12-08-00", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/K562/K562_dynamic_test.pth",
			"label":"Dynamic region"},
		{"type":"model","target": "K562", "model_type": "gru",\
			"run":"run_2021-05-28-12-08-00", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/K562/K562_static_test.pth",
			"label":"Static region"},
		{"type":"mean_baseline","target":"K562","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/K562_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"mean_by_region","target":"K562","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/K562_dynamic_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/K562_dynamic_evec.txt",\
		"label":"Dynamic region (mean)"},
		{"type":"mean_by_region","target":"K562","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/K562_static_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/K562_static_evec.txt",\
		"label":"Static region (mean)"}
		],

		18:[
		"NHEK",
		"test",
		"NHEK Testing, res: 100kb",
		{"type":"model","target": "NHEK", "model_type": "gru",\
			"run":"run_2021-05-28-04-10-01", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/NHEK/5_fold/test.pth",
			"label":"All"},
		{"type":"model","target": "NHEK", "model_type": "gru",\
			"run":"run_2021-05-28-04-10-01", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/NHEK/NHEK_dynamic_test.pth",
			"label":"Dynamic region"},
		{"type":"model","target": "NHEK", "model_type": "gru",\
			"run":"run_2021-05-28-04-10-01", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/NHEK/NHEK_static_test.pth",
			"label":"Static region"},
		{"type":"mean_baseline","target":"NHEK","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/NHEK_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"mean_by_region","target":"NHEK","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/NHEK_dynamic_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/NHEK_dynamic_evec.txt",\
		"label":"Dynamic region (mean)"},
		{"type":"mean_by_region","target":"NHEK","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/NHEK_static_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/NHEK_static_evec.txt",\
		"label":"Static region (mean)"}
		],

		19:[
		"HMEC",
		"test",
		"HMEC Testing, res: 100kb",
		{"type":"model","target": "HMEC", "model_type": "gru",\
			"run":"run_2021-05-28-08-25-27", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/HMEC/5_fold/test.pth",
			"label":"All"},
		{"type":"model","target": "HMEC", "model_type": "gru",\
			"run":"run_2021-05-28-08-25-27", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/HMEC/HMEC_dynamic_test.pth",
			"label":"Dynamic region"},
		{"type":"model","target": "HMEC", "model_type": "gru",\
			"run":"run_2021-05-28-08-25-27", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/HMEC/HMEC_static_test.pth",
			"label":"Static region"},
		{"type":"mean_baseline","target":"HMEC","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/HMEC_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"mean_by_region","target":"HMEC","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/HMEC_dynamic_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/HMEC_dynamic_evec.txt",\
		"label":"Dynamic region (mean)"},
		{"type":"mean_by_region","target":"HMEC","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/HMEC_static_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/HMEC_static_evec.txt",\
		"label":"Static region (mean)"}
		],

		20:[
		"HUVEC",
		"test",
		"HUVEC Testing, res: 100kb",
		{"type":"model","target": "HUVEC", "model_type": "gru",\
			"run":"run_2021-05-28-20-15-10", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/HUVEC/5_fold/test.pth",
			"label":"All"},
		{"type":"model","target": "HUVEC", "model_type": "gru",\
			"run":"run_2021-05-28-20-15-10", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/HUVEC/HUVEC_dynamic_test.pth",
			"label":"Dynamic region"},
		{"type":"model","target": "HUVEC", "model_type": "gru",\
			"run":"run_2021-05-28-20-15-10", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/HUVEC/HUVEC_static_test.pth",
			"label":"Static region"},
		{"type":"mean_baseline","target":"HUVEC","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/HUVEC_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"mean_by_region","target":"HUVEC","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/HUVEC_dynamic_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/HUVEC_dynamic_evec.txt",\
		"label":"Dynamic region (mean)"},
		{"type":"mean_by_region","target":"HUVEC","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/HUVEC_static_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/HUVEC_static_evec.txt",\
		"label":"Static region (mean)"}
		],
		21:[
		"IMR90",
		"test",
		"IMR90 Testing, res: 100kb",
		{"type":"model","target": "IMR90", "model_type": "gru",\
			"run":"run_2021-07-04-16-59-26", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/IMR90/5_fold/test.pth",
			"label":"All"},
		{"type":"model","target": "IMR90", "model_type": "gru",\
			"run":"run_2021-07-04-16-59-26", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/IMR90/IMR90_dynamic_test.pth",
			"label":"Dynamic region"},
		{"type":"model","target": "IMR90", "model_type": "gru",\
			"run":"run_2021-07-04-16-59-26", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/region_test/IMR90/IMR90_static_test.pth",
			"label":"Static region"},
		{"type":"mean_baseline","target":"IMR90","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/IMR90_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"mean_by_region","target":"IMR90","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/IMR90_dynamic_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/IMR90_dynamic_evec.txt",\
		"label":"Dynamic region (mean)"},
		{"type":"mean_by_region","target":"IMR90","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/IMR90_static_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/IMR90_static_evec.txt",\
		"label":"Static region (mean)"}
		],
		# mean eigenvector
		22:[
		"GM12878",
		"test",
		"GM12878 Testing, res: 100kb",
		{"type":"mean_baseline","target":"GM12878","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/GM12878_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"mean_by_region","target":"GM12878","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/GM12878_dynamic_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/GM12878_dynamic_evec.txt",\
		"label":"Dynamic region (mean)"},
		{"type":"mean_by_region","target":"GM12878","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/GM12878_static_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/GM12878_static_evec.txt",\
		"label":"Static region (mean)"}
		],

		23:[
		"K562",
		"test",
		"K562 Testing, res: 100kb",
		{"type":"mean_baseline","target":"K562","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/K562_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"mean_by_region","target":"K562","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/K562_dynamic_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/K562_dynamic_evec.txt",\
		"label":"Dynamic region (mean)"},
		{"type":"mean_by_region","target":"K562","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/K562_static_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/K562_static_evec.txt",\
		"label":"Static region (mean)"}
		],

		24:[
		"NHEK",
		"test",
		"NHEK Testing, res: 100kb",
		{"type":"mean_baseline","target":"NHEK","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/NHEK_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"mean_by_region","target":"NHEK","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/NHEK_dynamic_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/NHEK_dynamic_evec.txt",\
		"label":"Dynamic region (mean)"},
		{"type":"mean_by_region","target":"NHEK","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/NHEK_static_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/NHEK_static_evec.txt",\
		"label":"Static region (mean)"}
		],

		25:[
		"HMEC",
		"test",
		"HMEC Testing, res: 100kb",
		{"type":"mean_baseline","target":"HMEC","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/HMEC_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"mean_by_region","target":"HMEC","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/HMEC_dynamic_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/HMEC_dynamic_evec.txt",\
		"label":"Dynamic region (mean)"},
		{"type":"mean_by_region","target":"HMEC","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/HMEC_static_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/HMEC_static_evec.txt",\
		"label":"Static region (mean)"}
		],

		26:[
		"HUVEC",
		"test",
		"HUVEC Testing, res: 100kb",
		{"type":"mean_baseline","target":"HUVEC","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/HUVEC_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"mean_by_region","target":"HUVEC","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/HUVEC_dynamic_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/HUVEC_dynamic_evec.txt",\
		"label":"Dynamic region (mean)"},
		{"type":"mean_by_region","target":"HUVEC","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/HUVEC_static_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/HUVEC_static_evec.txt",\
		"label":"Static region (mean)"}
		],
		27:[
		"IMR90",
		"test",
		"IMR90 Testing, res: 100kb",
		{"type":"mean_baseline","target":"IMR90","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/IMR90_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"mean_by_region","target":"IMR90","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/IMR90_dynamic_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/IMR90_dynamic_evec.txt",\
		"label":"Dynamic region (mean)"},
		{"type":"mean_by_region","target":"IMR90","resolution":"100kb",\
		"predict_dir":"./data/mean_evec/mean_evec_by_region_100kb/IMR90_static_mean.txt",\
		"target_dir":"./data/mean_evec/mean_evec_by_region_100kb/IMR90_static_evec.txt",\
		"label":"Static region (mean)"}
		],

		#50kb no NHEK - best model in validayion
		28:[
		"GM12878",
		"test",
		"GM12878 Testing, res: 100kb",
		{"type":"model","target": "GM12878", "model_type": "gru",\
			"run":"run_2021-06-18-14-33-36", "resolution":"50kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_50kb/processed/cross_validation/5_cells/IMR90-HMEC-K562-GM12878-HUVEC/GM12878/5_fold/test.pth",
			"label":"gru (all, no NHEK)"},
		{"type":"mean_baseline","target":"GM12878","resolution":"50kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_50kb_allchromosome_no_NHEK/GM12878_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		29:[
		"K562",
		"test",
		"K562 Testing, res: 100kb",
		{"type":"model","target": "K562", "model_type": "gru",\
			"run":"run_2021-06-20-10-53-10", "resolution":"50kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_50kb/processed/cross_validation/5_cells/IMR90-HMEC-K562-GM12878-HUVEC/K562/5_fold/test.pth",
			"label":"gru (all, no NHEK)"},
		{"type":"mean_baseline","target":"K562","resolution":"50kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_50kb_allchromosome_no_NHEK/K562_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		30:[
		"HMEC",
		"test",
		"HMEC Testing, res: 100kb",
		{"type":"model","target": "HMEC", "model_type": "gru",\
			"run":"run_2021-06-20-04-50-30", "resolution":"50kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_50kb/processed/cross_validation/5_cells/IMR90-HMEC-K562-GM12878-HUVEC/HMEC/5_fold/test.pth",
			"label":"gru (all, no NHEK)"},
		{"type":"mean_baseline","target":"HMEC","resolution":"50kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_50kb_allchromosome_no_NHEK/HMEC_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		31:[
		"HUVEC",
		"test",
		"HUVEC Testing, res: 100kb",
		{"type":"model","target": "HUVEC", "model_type": "gru",\
			"run":"run_2021-06-21-03-29-33", "resolution":"50kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_50kb/processed/cross_validation/5_cells/IMR90-HMEC-K562-GM12878-HUVEC/HUVEC/5_fold/test.pth",
			"label":"gru (all, no NHEK)"},
		{"type":"mean_baseline","target":"HUVEC","resolution":"50kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_50kb_allchromosome_no_NHEK/HUVEC_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		32:[
		"IMR90",
		"test",
		"IMR90 Testing, res: 100kb",
		{"type":"model","target": "IMR90", "model_type": "gru",\
			"run":"run_2021-06-19-16-09-56", "resolution":"50kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_50kb/processed/cross_validation/5_cells/IMR90-HMEC-K562-GM12878-HUVEC/IMR90/5_fold/test.pth",
			"label":"gru (all, no NHEK)"},
		{"type":"mean_baseline","target":"IMR90","resolution":"50kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_50kb_allchromosome_no_NHEK/IMR90_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],
		#100 kb add mean
		33:[
		"GM12878",
		"test",
		"GM12878 Testing, res: 100kb",
		{"type":"model","target": "GM12878", "model_type": "gru",\
			"run":"run_2021-06-21-23-11-40", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation_with_mean/GM12878/5_fold/test.pth",
			"label":"gru (all, with mean)"},
		{"type":"mean_baseline","target":"GM12878","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/GM12878_test.txt",
			"label":"mean_evec_3cells","split":"cross_validation"}
		],
		34:[
		"IMR90",
		"test",
		"IMR90 Testing, res: 100kb",
		{"type":"model","target": "IMR90", "model_type": "gru",\
			"run":"run_2021-06-21-22-00-34", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation_with_mean/IMR90/5_fold/test.pth",
			"label":"gru (all, with mean)"},
		{"type":"mean_baseline","target":"IMR90","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/IMR90_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],
		35:[
		"HMEC",
		"test",
		"IMR90 Testing, res: 100kb",
		{"type":"model","target": "HMEC", "model_type": "gru",\
			"run":"run_2021-06-22-12-43-26", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation_with_mean/HMEC/5_fold/test.pth",
			"label":"gru (all, with mean)"},
		{"type":"mean_baseline","target":"HMEC","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/HMEC_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		36:[
		"K562",
		"test",
		"K562 Testing, res: 100kb",
		{"type":"model","target": "K562", "model_type": "gru",\
			"run":"run_2021-06-22-18-31-50", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation_with_mean/K562/5_fold/test.pth",
			"label":"gru (all, with mean)"},
		{"type":"mean_baseline","target":"K562","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/K562_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],
		37:[
		"HUVEC",
		"test",
		"K562 Testing, res: 100kb",
		{"type":"model","target": "HUVEC", "model_type": "gru",\
			"run":"run_2021-06-23-01-21-40", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation_with_mean/HUVEC/5_fold/test.pth",
			"label":"gru (all, with mean)"},
		{"type":"mean_baseline","target":"HUVEC","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/HUVEC_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		38:[
		"NHEK",
		"test",
		"K562 Testing, res: 100kb",
		{"type":"model","target": "NHEK", "model_type": "gru",\
			"run":"run_2021-06-22-10-07-57", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation_with_mean/NHEK/5_fold/0/train.pth",
			"label":"gru (all, with mean)"},
		{"type":"mean_baseline","target":"NHEK","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/NHEK_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],
		#50kb no NHEK -worst model in validation
		39:[
		"GM12878",
		"test",
		"GM12878 Testing, res: 100kb",
		{"type":"model","target": "GM12878", "model_type": "gru",\
			"run":"run_2021-06-19-12-04-10", "resolution":"50kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_50kb/processed/cross_validation/5_cells/IMR90-HMEC-K562-GM12878-HUVEC/GM12878/5_fold/test.pth",
			"label":"gru (all, no NHEK)"},
		{"type":"mean_baseline","target":"GM12878","resolution":"50kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_50kb_allchromosome_no_NHEK/GM12878_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		40:[
		"K562",
		"test",
		"K562 Testing, res: 100kb",
		{"type":"model","target": "K562", "model_type": "gru",\
			"run":"run_2021-06-20-09-18-12", "resolution":"50kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_50kb/processed/cross_validation/5_cells/IMR90-HMEC-K562-GM12878-HUVEC/K562/5_fold/test.pth",
			"label":"gru (all, no NHEK)"},
		{"type":"mean_baseline","target":"K562","resolution":"50kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_50kb_allchromosome_no_NHEK/K562_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		41:[
		"HMEC",
		"test",
		"HMEC Testing, res: 100kb",
		{"type":"model","target": "HMEC", "model_type": "gru",\
			"run":"run_2021-06-19-23-15-14", "resolution":"50kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_50kb/processed/cross_validation/5_cells/IMR90-HMEC-K562-GM12878-HUVEC/HMEC/5_fold/test.pth",
			"label":"gru (all, no NHEK)"},
		{"type":"mean_baseline","target":"HMEC","resolution":"50kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_50kb_allchromosome_no_NHEK/HMEC_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		42:[
		"HUVEC",
		"test",
		"HUVEC Testing, res: 100kb",
		{"type":"model","target": "HUVEC", "model_type": "gru",\
			"run":"run_2021-06-20-21-58-06", "resolution":"50kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_50kb/processed/cross_validation/5_cells/IMR90-HMEC-K562-GM12878-HUVEC/HUVEC/5_fold/test.pth",
			"label":"gru (all, no NHEK)"},
		{"type":"mean_baseline","target":"HUVEC","resolution":"50kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_50kb_allchromosome_no_NHEK/HUVEC_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		43:[
		"IMR90",
		"test",
		"IMR90 Testing, res: 100kb",
		{"type":"model","target": "IMR90", "model_type": "gru",\
			"run":"run_2021-06-19-13-19-08", "resolution":"50kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_50kb/processed/cross_validation/5_cells/IMR90-HMEC-K562-GM12878-HUVEC/IMR90/5_fold/test.pth",
			"label":"gru (all, no NHEK)"},
		{"type":"mean_baseline","target":"IMR90","resolution":"50kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_50kb_allchromosome_no_NHEK/IMR90_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],
		
		# updated add mean
		44:[
		"NHEK",
		"test",
		"NHEK Testing, res: 100kb",
		{"type":"model","target": "NHEK", "model_type": "gru",\
			"run":"run_2021-06-30-13-44-11", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation_with_mean/NHEK/5_fold/test.pth",
			"label":"gru (all, with mean)"},
		{"type":"mean_baseline","target":"NHEK","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/NHEK_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		45:[
		"GM12878",
		"test",
		"GM12878 Testing, res: 100kb",
		{"type":"model","target": "GM12878", "model_type": "gru",\
			"run":"run_2021-07-03-03-37-38", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation_with_mean/GM12878/5_fold/test.pth",
			"label":"gru (all, with mean)"},
		{"type":"mean_baseline","target":"GM12878","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/GM12878_test.txt",
			"label":"mean_evec_3cells","split":"cross_validation"}
		],
		46:[
		"IMR90",
		"test",
		"IMR90 Testing, res: 100kb",
		{"type":"model","target": "IMR90", "model_type": "gru",\
			"run":"run_2021-07-02-06-41-41", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation_with_mean/IMR90/5_fold/test.pth",
			"label":"gru (all, with mean)"},
		{"type":"mean_baseline","target":"IMR90","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/IMR90_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],
		47:[
		"HMEC",
		"test",
		"HMEC Testing, res: 100kb",
		{"type":"model","target": "HMEC", "model_type": "gru",\
			"run":"run_2021-07-01-08-17-50", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation_with_mean/HMEC/5_fold/test.pth",
			"label":"gru (all, with mean)"},
		{"type":"mean_baseline","target":"HMEC","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/HMEC_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		48:[
		"K562",
		"test",
		"K562 Testing, res: 100kb",
		{"type":"model","target": "K562", "model_type": "gru",\
			"run":"run_2021-07-03-15-15-20", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation_with_mean/K562/5_fold/test.pth",
			"label":"gru (all, with mean)"},
		{"type":"mean_baseline","target":"K562","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/K562_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],
		49:[
		"HUVEC",
		"test",
		"HUVEC Testing, res: 100kb",
		{"type":"model","target": "HUVEC", "model_type": "gru",\
			"run":"run_2021-06-30-21-48-21", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation_with_mean/HUVEC/5_fold/test.pth",
			"label":"gru (all, with mean)"},
		{"type":"mean_baseline","target":"HUVEC","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/HUVEC_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		# test by 6 region
		50:[
		"IMR90",
		"test",
		"IMR90 Testing, res: 100kb",
		{"type":"model","target": "IMR90", "model_type": "gru",\
			"run":"run_2021-07-04-16-59-26", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/IMR90/5_fold/test.pth",
			"label":"All"},
		{"type":"mean_baseline","target":"IMR90","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/IMR90_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"model_by_region","target": "IMR90", "model_type": "gru",\
			"run":"run_2021-07-04-16-59-26", "resolution":"100kb",\
			"number_of_region":6,
			"dataloader_dir":"./data/mean_evec/mean_evec_by_region_100kb/GM12878",
			"label":""},
		],

		51:[
		"IMR90",
		"test",
		"IMR90 Testing, res: 100kb",
		{"type":"model","target": "IMR90", "model_type": "gru",\
			"run":"run_2021-07-04-16-59-26", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/IMR90/5_fold/test.pth",
			"label":"All"},
		{"type":"mean_baseline","target":"IMR90","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/IMR90_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"model_by_region","target": "IMR90", "model_type": "gru",\
			"run":"run_2021-07-04-16-59-26", "resolution":"100kb",\
			"number_of_region":6,
			"dataloader_dir":"./data/mean_evec/mean_evec_by_region_100kb/IMR90",
			"label":""},
		{"type":"mean_by_multiple_region","target":"IMR90","resolution":"100kb",\
			"number_of_region":6,
			"data_dir":"./data/mean_evec/mean_evec_by_region_100kb/IMR90"},
		],
		52:[
		"GM12878",
		"test",
		"GM12878 Testing, res: 100kb",
		{"type":"model","target": "GM12878", "model_type": "gru",\
			"run":"run_2021-05-27-14-59-14", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/GM12878/5_fold/test.pth",
			"label":"All"},
		{"type":"mean_baseline","target":"GM12878","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/GM12878_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"model_by_region","target": "GM12878", "model_type": "gru",\
			"run":"run_2021-05-27-14-59-14", "resolution":"100kb",\
			"number_of_region":6,
			"dataloader_dir":"./data/mean_evec/mean_evec_by_region_100kb/GM12878",
			"label":""},
		{"type":"mean_by_multiple_region","target":"GM12878","resolution":"100kb",\
			"number_of_region":6,
			"data_dir":"./data/mean_evec/mean_evec_by_region_100kb/GM12878"},
		],
		53:[
		"K562",
		"test",
		"K562 Testing, res: 100kb",
		{"type":"model","target": "K562", "model_type": "gru",\
			"run":"run_2021-05-28-12-08-00", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/K562/5_fold/test.pth",
			"label":"All"},
		{"type":"mean_baseline","target":"K562","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/K562_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"model_by_region","target": "K562", "model_type": "gru",\
			"run":"run_2021-05-28-12-08-00", "resolution":"100kb",\
			"number_of_region":6,
			"dataloader_dir":"./data/mean_evec/mean_evec_by_region_100kb/K562",
			"label":""},
		{"type":"mean_by_multiple_region","target":"K562","resolution":"100kb",\
			"number_of_region":6,
			"data_dir":"./data/mean_evec/mean_evec_by_region_100kb/K562"},
		],
		54:[
		"NHEK",
		"test",
		"NHEK Testing, res: 100kb",
		{"type":"model","target": "NHEK", "model_type": "gru",\
			"run":"run_2021-05-28-04-10-01", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/NHEK/5_fold/test.pth",
			"label":"All"},
		{"type":"mean_baseline","target":"NHEK","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/NHEK_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"model_by_region","target": "NHEK", "model_type": "gru",\
			"run":"run_2021-05-28-04-10-01", "resolution":"100kb",\
			"number_of_region":6,
			"dataloader_dir":"./data/mean_evec/mean_evec_by_region_100kb/NHEK",
			"label":""},
		{"type":"mean_by_multiple_region","target":"NHEK","resolution":"100kb",\
			"number_of_region":6,
			"data_dir":"./data/mean_evec/mean_evec_by_region_100kb/NHEK"},
		],
		55:[
		"HMEC",
		"test",
		"HMEC Testing, res: 100kb",
		{"type":"model","target": "HMEC", "model_type": "gru",\
			"run":"run_2021-05-28-08-25-27", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/HMEC/5_fold/test.pth",
			"label":"All"},
		{"type":"mean_baseline","target":"HMEC","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/HMEC_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"model_by_region","target": "HMEC", "model_type": "gru",\
			"run":"run_2021-05-28-08-25-27", "resolution":"100kb",\
			"number_of_region":6,
			"dataloader_dir":"./data/mean_evec/mean_evec_by_region_100kb/HMEC",
			"label":""},
		{"type":"mean_by_multiple_region","target":"HMEC","resolution":"100kb",\
			"number_of_region":6,
			"data_dir":"./data/mean_evec/mean_evec_by_region_100kb/HMEC"},
		],
		56:[
		"HUVEC",
		"test",
		"HUVEC Testing, res: 100kb",
		{"type":"model","target": "HUVEC", "model_type": "gru",\
			"run":"run_2021-05-28-20-15-10", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation/HUVEC/5_fold/test.pth",
			"label":"All"},
		{"type":"mean_baseline","target":"HUVEC","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/HUVEC_test.txt",
			"label":"All (mean)","split":"cross_validation"},
		{"type":"model_by_region","target": "HUVEC", "model_type": "gru",\
			"run":"run_2021-05-28-20-15-10", "resolution":"100kb",\
			"number_of_region":6,
			"dataloader_dir":"./data/mean_evec/mean_evec_by_region_100kb/HUVEC",
			"label":""},
		{"type":"mean_by_multiple_region","target":"HUVEC","resolution":"100kb",\
			"number_of_region":6,
			"data_dir":"./data/mean_evec/mean_evec_by_region_100kb/HUVEC"},
		],

		57:[
		"HUVEC",
		"test",
		"HUVEC Testing, res: 100kb",
		{"type":"model","target": "HUVEC", "model_type": "gru",\
			"run":"run_2021-07-01-00-52-29", "resolution":"100kb",\
			"dataloader_dir":"./data/6_cell_input_updated/6_cell_input_updated_100kb/processed/cross_validation_with_mean/HUVEC/5_fold/test.pth",
			"label":"gru (all, with mean)"},
		{"type":"mean_baseline","target":"HUVEC","resolution":"100kb",\
			"data_dir":"./data/mean_evec/mean_evec_updated_100kb_allchromosome/HUVEC_test.txt",
			"label":"mean_evec","split":"cross_validation"}
		],

		

	}