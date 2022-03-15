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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)





def test(model, test_loader,task):
	# print("===Testing...")
	model.eval()
	loss_fn = nn.CrossEntropyLoss()
	cur_loss = []
	cur_pred = []
	cur_labels = []
	for batch in tqdm(test_loader):
		inputs = batch["input"]
		inputs = inputs.to(device)
		if task == "cla":
			labels = batch["cla_labels"]
		elif task == "reg":
			labels = batch["reg_labels"]
		inputs = inputs.to(device)
		labels = labels.to(device)
		y_pred = model.forward(inputs)
		loss = loss_fn(y_pred, labels)
		loss = loss.cpu().detach().numpy()
		cur_loss.append(loss)
		cur_pred += y_pred[:,-1].detach().tolist()
		cur_labels += labels.detach().tolist()
	


	curve_name = "AUROC"
	auroc, fpr, tpr = AUROC(experiment,cur_labels,cur_pred, None, curve_name,save_fig = False)
	
	curve_name = "AUPRC"
	auprc, recall, precision = AUPRC(experiment,cur_labels,cur_pred, None, curve_name,save_fig = False)
	
	data = {}
	data["auroc"] = {}
	data["auroc"]["auroc_score"] = auroc
	data["auroc"]["fpr"] = fpr
	data["auroc"]["tpr"] = tpr

	data["auprc"] = {}
	data["auprc"]["auprc_score"] = auprc
	data["auprc"]["recall"] = recall
	data["auprc"]["precision"] = precision
	
	return data


def test_by_region(model,test_loader,task):
	# print("===Testing...")
	model.eval()
	loss_fn = nn.CrossEntropyLoss()
	cur_loss = []
	cur_pred = []
	cur_labels = []
	for batch in tqdm(test_loader):
		inputs = batch["input"]
		inputs = inputs.to(device)
		if task == "cla":
			labels = batch["cla_labels"]
		elif task == "reg":
			labels = batch["reg_labels"]
		inputs = inputs.to(device)
		labels = labels.to(device)
		y_pred = model.forward(inputs)
		loss = loss_fn(y_pred, labels)
		loss = loss.cpu().detach().numpy()
		cur_loss.append(loss)
		cur_pred += y_pred[:,-1].detach().tolist()
		cur_labels += labels.detach().tolist()
	


	curve_name = "AUROC"
	auroc, fpr, tpr = AUROC(experiment,cur_labels,cur_pred, None, curve_name,save_fig = False)
	
	curve_name = "AUPRC"
	auprc, recall, precision = AUPRC(experiment,cur_labels,cur_pred, None, curve_name,save_fig = False)
	
	data = {}
	data["auroc"] = {}
	data["auroc"]["auroc_score"] = auroc
	data["auroc"]["fpr"] = fpr
	data["auroc"]["tpr"] = tpr

	data["auprc"] = {}
	data["auprc"]["auprc_score"] = auprc
	data["auprc"]["recall"] = recall
	data["auprc"]["precision"] = precision
	
	return data

def combine_plot_curve(data,exp_dpath):
	# data_dpath = os.path.join("data","exp_data","curve_info.json")
	# data = {}
	# if os.path.isfile(data_dpath):
	# 	with open(data_dpath, 'r') as j:
	# 		data = json.loads(j.read())

	

	#plot auroc curve
	
	#pyplot.rcParams['font.size'] = '20'
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

def combine_plot_curve_with_target(data,exp_dpath,target,do_test):
	# data_dpath = os.path.join("data","exp_data","curve_info.json")
	# data = {}
	# if os.path.isfile(data_dpath):
	# 	with open(data_dpath, 'r') as j:
	# 		data = json.loads(j.read())

	

	#plot auroc curve
	if do_test:
		pyplot.title("{} AUROC (target cell: {})".format("Test",target))
	else:
		pyplot.title("{} AUROC (target cell: {})".format("Validation",target))
	img_path = os.path.join(exp_dpath,"{}_auroc.png".format(target))

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
	if do_test:
		pyplot.title("{} AUPRC (target cell: {})".format("Test",target))
	else:
		pyplot.title("{} AUPRC (target cell: {})".format("Validation",target))
	img_path = os.path.join(exp_dpath,"{}_auprc.png".format(target))
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
	parser = argparse.ArgumentParser()
	parser.add_argument("-b", "--baseline", action="store_true",
                        help="add baseline")
	parser.add_argument("-t", "--test", action="store_true",
                        help="add baseline")
	parser.add_argument(
    	"--cell", default=None, type=str, help="the cell line for validation and testing",
    )
	parser.add_argument("--model", default = None, type = str,
                        help="model type")
	parser.add_argument("--tag", default = None, type = str,
                        help="tag for curve")

	parser.add_argument("--exp", default = None, type = str,
                        help="the exp number ")


	parser.add_argument("--resolution", default = "50kb", type = str,
                        help="the resolution of eigenvectors")

	parser.add_argument("--split", default = "5_cells", type = str,
                        help="the resolution of eigenvectors")



	# parser.add_argument('-m','--list', nargs='+', help='list of model type')
	# parser.add_argument('-run','--list', nargs='+', help='None')
	# parser.add_argument('-t','--list', nargs='+', help='list of model name')
	args = parser.parse_args()
	
	

	# models = ["cnn_5","cnn_10","cnn_3"]
	# run = ["run_2021-03-16-22-01-35","run_2021-03-16-18-29-07","run_2021-03-16-22-45-15"]
	# tags = ["cnn_5","cnn_10","IMR90"]

	# models = ["cnn_3","cnn_6"]
	# run = ["run_2021-03-16-22-45-15","run_2021-03-17-00-36-12"]
	# tags = ["IMR90","6 cell lines"]

	# models = ["cnn_3","cnn_5","cnn_6","cnn_7","cnn_8","cnn_10"]
	# run = ["run_2021-03-16-22-45-15","run_2021-03-16-22-01-35","run_2021-03-17-00-36-12", "run_2021-03-17-10-37-54","run_2021-03-17-10-13-43","run_2021-03-16-18-29-07"]
	# tags = ["IMR90","cnn_5","cnn_6","cnn_7","cnn_8","cnn_10"]


	# models = ["cnn_3","cnn_5","cnn_6","cnn_7","cnn_8","cnn_9","cnn_10"]
	# run = ["run_2021-03-16-22-45-15","run_2021-03-16-22-01-35","run_2021-03-17-00-36-12", "run_2021-03-17-10-37-54","run_2021-03-17-10-13-43","run_2021-03-17-11-28-54","run_2021-03-17-11-02-59"]
	# tags = ["IMR90","cnn_5","cnn_6","cnn_7","cnn_8","cnn_9","cnn_10"]

	# models  = ["cnn_3","cnn_8"]
	# run = ["run_2021-03-16-22-45-15","run_2021-03-17-10-13-43"]	
	# tags = ["IMR90", "6_cell_lines"]
	do_test = args.test

	now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	exp_dpath = os.path.join("data","exp_data","combined_curve","exp_{}".format(now))
	if not os.path.exists(exp_dpath):
		os.makedirs(exp_dpath)

	print(exp_dpath)

	experiment.log_text(now)`
	#target = "IMR90"
	#models = []
	#run = []
	#tags = []
	#gnerate_curve(target,models,run,tags,do_test,exp_dpath,args)


	#target = "IMR90"
	#models = ["cnn_3"]
	#run = ["run_2021-03-25-15-23-09"]
	#tags = ["5_cells"]
	#gnerate_curve(target,models,run,tags,do_test,exp_dpath,args)

	#target = "K562"
	#models = ["cnn_8"]
	#run = ["run_2021-03-24-15-27-10"]
	#tags = ["5_cells"]
	#gnerate_curve(target,models,run,tags,do_test,exp_dpath,args)

	# target = "IMR90"
	# models = ["cnn_8","cnn_3"]
	# run = ["run_2021-03-29-20-51-16","run_2021-03-25-17-10-14"]
	# tags = ["5_cells","IMR90"]
	#gnerate_curve(target,models,run,tags,do_test,exp_dpath,args)


	# ===========================================================
	# Version： results for 6 cells. cnn_8 (March 24 2021)
	# ===========================================================
	# target = "IMR90"
	# models = ["cnn_8","cnn_3","lstm","lstm"]
	# run = ["run_2021-04-18-18-33-08","run_2021-04-19-00-23-59",\
	# "run_2021-04-26-10-09-13","run_2021-04-26-10-17-11"]
	# tags = ["cnn_5cell","cnn_single","lstm_5cell","lstm_single"]
	# gnerate_curve(target,models,run,tags,do_test,exp_dpath,args)

	# target = "IMR90"
	# models = ["gru","gru"]
	# run = ["run_2021-04-28-06-05-35","run_2021-04-27-22-32-00"]
	# tags = ["gru_5cell","gru_single"]
	# gnerate_curve(target,models,run,tags,do_test,exp_dpath,args)

	# target = "GM12878"
	# models = ["cnn_8","cnn_3"]
	# run = ["run_2021-04-21-09-44-18","run_2021-04-19-17-00-04"]
	# tags = ["5_cells","GM12878"]
	# gnerate_curve(target,models,run,tags,do_test,exp_dpath,args)

	#target = "HUVEC"
	#models = ["cnn_8","cnn_3"]
	#run = ["run_2021-04-20-09-53-43","run_2021-04-19-20-42-35"]
	#tags = ["5_cells","HUVEC"]
	#gnerate_curve(target,models,run,tags,do_test,exp_dpath,args)

	#target = "K562"
	#models = ["cnn_8","cnn_3"]
	#run = ["run_2021-04-19-22-57-10","run_2021-04-19-20-00-02"]
	#tags = ["5_cells","K562"]
	#gnerate_curve(target,models,run,tags,do_test,exp_dpath,args)

	#target = "NHEK"
	#models = ["cnn_8","cnn_3"]
	#run = ["run_2021-04-20-00-30-20","run_2021-04-19-20-22-01"]
	#tags = ["5_cells","NHEK"]
	#gnerate_curve(target,models,run,tags,do_test,exp_dpath,args)

	#target = "HMEC"
	#models = ["cnn_8","cnn_3"]
	#run = ["run_2021-04-20-01-07-10","run_2021-04-19-20-33-21"]
	#tags = ["5_cells","HMEC"]
	#gnerate_curve(target,models,run,tags,do_test,exp_dpath,args)


	#target = args.cell
	#models = [args.model]
	#run = ["run_{}".format(args.exp)]
	#tags = [args.tag]
	#gnerate_curve(target,models,run,tags,do_test,exp_dpath,args)

	# ==========================================================
	# END
	# ==========================================================

	# ==========================================================
	# 10kb vs 50kb CNN 
	# ==========================================================


	target = "IMR90"
	models = ["cnn_8","cnn_8"]
	run = ["run_2021-03-24-09-09-38","run_2021-04-18-18-33-08"]
	tags = ["cnn_100kb","cnn_50kb"]
	resolutions = ["100kb","50kb"]
	gnerate_curve(target,models,run,tags,resolutions, do_test,exp_dpath,args)

	target = "GM12878"
	models = ["cnn_8","cnn_8"]
	run = ["run_2021-03-24-14-19-33","run_2021-04-19-21-10-28"]
	tags = ["cnn_100kb","cnn_50kb"]
	resolution = ["100kb","50kb"]
	gnerate_curve(target,models,run,tags,resolution, do_test,exp_dpath,args)

	target = "HUVEC"
	models = ["cnn_8","cnn_8"]
	run = ["run_2021-03-25-16-01-11","run_2021-04-20-09-53-43"]
	tags = ["cnn_100kb","cnn_50kb"]
	resolution = ["100kb","50kb"]
	gnerate_curve(target,models,run,tags,resolution, do_test,exp_dpath,args)

	target = "K562"
	models = ["cnn_8","cnn_8"]
	run = ["run_2021-03-24-15-27-10","run_2021-04-19-22-57-10"]
	tags = ["cnn_100kb","cnn_50kb"]
	resolution = ["100kb","50kb"]
	gnerate_curve(target,models,run,tags,resolution, do_test,exp_dpath,args)

	target = "NHEK"
	models = ["cnn_8","cnn_8"]
	run = ["run_2021-03-25-16-28-38","run_2021-04-20-00-30-20"]
	tags = ["cnn_100kb","cnn_50kb"]
	resolution = ["100kb","50kb"]
	gnerate_curve(target,models,run,tags,resolution, do_test,exp_dpath,args)

	target = "HMEC"
	models = ["cnn_8","cnn_8"]
	run = ["run_2021-03-24-15-51-43","run_2021-04-20-01-07-10"]
	tags = ["cnn_100kb","cnn_50kb"]
	resolution = ["100kb","50kb"]
	gnerate_curve(target,models,run,tags,resolution, do_test,exp_dpath,args)

	

	# ==========================================================
	# 50kb CNN v.s GRU
	# ==========================================================


	# target = "IMR90"
	# models = ["cnn_8","gru"]
	# run = ["run_2021-04-18-18-33-08","run_2021-04-27-22-32-00"]
	# tags = ["cnn","gru"]
	# resolutions = ["50kb","50kb"]
	# gnerate_curve(target,models,run,tags,resolutions, do_test,exp_dpath,args)

	# target = "GM12878"
	# models = ["cnn_8","gru"]
	# run = ["run_2021-04-19-21-10-28","run_2021-05-01-04-35-31"]
	# tags = ["cnn","gru"]
	# resolutions = ["50kb","50kb"]
	# gnerate_curve(target,models,run,tags,resolutions, do_test,exp_dpath,args)

	# target = "HUVEC"
	# models = ["cnn_8","gru"]
	# run = ["run_2021-04-20-09-53-43","run_2021-05-01-15-58-30"]
	# tags = ["cnn","gru"]
	# resolutions = ["50kb","50kb"]
	# gnerate_curve(target,models,run,tags,resolutions, do_test,exp_dpath,args)

	# target = "K562"
	# models = ["cnn_8","gru"]
	# run = ["run_2021-04-19-22-57-10","run_2021-04-30-13-52-06"]
	# tags = ["cnn","gru"]
	# resolutions = ["50kb","50kb"]
	# gnerate_curve(target,models,run,tags,resolutions, do_test,exp_dpath,args)

	# target = "NHEK"
	# models = ["cnn_8","gru"]
	# run = ["run_2021-04-20-00-30-20","run_2021-04-29-13-08-47"]
	# tags = ["cnn","gru"]
	# resolutions = ["50kb","50kb"]
	# gnerate_curve(target,models,run,tags,resolutions, do_test,exp_dpath,args)

	# target = "HMEC"
	# models = ["cnn_8","gru"]
	# run = ["run_2021-04-20-01-07-10","run_2021-04-29-22-40-23"]
	# tags = ["cnn","gru"]
	# resolutions = ["50kb","50kb"]
	# gnerate_curve(target,models,run,tags,resolutions, do_test,exp_dpath,args)





	# ==========================================================
	# 50kb bidirectional GRU vs one-direction GRU
	# ==========================================================


	# target = "IMR90"
	# models = ["gru","gru"]
	# run = ["run_2021-05-14-13-56-19","run_2021-04-27-22-32-00"]
	# tags = ["bi-gru","gru"]
	# resolutions = ["50kb","50kb"]
	# gnerate_curve(target,models,run,tags,resolutions, do_test,exp_dpath,args)

	# target = "GM12878"
	# models = ["gru","gru"]
	# run = ["run_2021-05-17-03-11-28","run_2021-05-01-04-35-31"]
	# tags = ["bi-gru","gru"]
	# resolutions = ["50kb","50kb"]
	# gnerate_curve(target,models,run,tags,resolutions, do_test,exp_dpath,args)

	# target = "HUVEC"
	# models = ["gru","gru"]
	# run = ["run_2021-05-17-15-13-02","run_2021-05-01-15-58-30"]
	# tags = ["bi-gru","gru"]
	# resolutions = ["50kb","50kb"]
	# gnerate_curve(target,models,run,tags,resolutions, do_test,exp_dpath,args)

	# target = "K562"
	# models = ["gru","gru"]
	# run = ["run_2021-05-15-20-37-01","run_2021-04-30-13-52-06"]
	# tags = ["bi-gru","gru"]
	# resolutions = ["50kb","50kb"]
	# gnerate_curve(target,models,run,tags,resolutions, do_test,exp_dpath,args)

	# target = "NHEK"
	# models = ["gru","gru"]
	# run = ["run_2021-05-15-02-00-59","run_2021-04-29-13-08-47"]
	# tags = ["bi-gru","gru"]
	# resolutions = ["50kb","50kb"]
	# gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)

	# target = "HMEC"
	# models = ["gru","gru"]
	# run = ["run_2021-05-15-05-26-58","run_2021-04-29-22-40-23"]
	# tags = ["bi-gru","gru"]
	# resolutions = ["50kb","50kb"]
	# gnerate_curve(target,models,run,tags,resolutions, do_test,exp_dpath,args)


	##=========================================
	## bi-gru + cross validation
	##=========================================

	# target = "NHEK"
	# models = ["gru"]
	# run = ["run_2021-05-18-02-23-02"]
	# tags = ["gru (all chromosomes)"]
	# resolutions = ["50kb"]
	# gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)

	# target = "IMR90"
	# models = ["gru"]
	# run = ["run_2021-05-17-03-34-10"]
	# tags = ["gru (all chromosomes)"]
	# resolutions = ["50kb"]
	# gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)
	##=========================================
	## 50 kb all chromosome gru + cross validation
	##=========================================

	# target = "GM12878"
	# models = ["gru"]
	# run = ["run_2021-05-18-20-46-47"]
	# tags = ["gru (all chromosomes)"]
	# resolutions = ["50kb"]
	# gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)

	# target = "K562"
	# models = ["gru"]
	# run = ["run_2021-05-23-00-10-46"]
	# tags = ["gru (all chromosomes)"]
	# resolutions = ["50kb"]
	# gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)

	# target = "HMEC"
	# models = ["gru"]
	# run = ["run_2021-05-21-11-41-58"]
	# tags = ["gru (all chromosomes)"]
	# resolutions = ["50kb"]
	# gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)

	# target = "HUVEC"
	# models = ["gru"]
	# run = ["run_2021-05-23-05-29-55"]
	# tags = ["gru (all chromosomes)"]
	# resolutions = ["50kb"]
	# gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)

	# target = "IMR90"
	# models = ["gru"]
	# run = ["run_2021-05-24-02-48-35"]
	# tags = ["gru (all chromosomes)"]
	# resolutions = ["50kb"]
	# gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)

	# target = "NHEK"
	# models = ["gru"]
	# run = ["run_2021-05-20-21-06-24"]
	# tags = ["gru (all chromosomes)"]
	# resolutions = ["50kb"]
	# gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)


	
	##=========================================
	## 100 kb all chromosome gru + cross validation
	##=========================================

	# target = "GM12878"
	# models = ["gru"]
	# run = ["run_2021-05-27-14-59-14"]
	# tags = ["gru (all chromosomes)"]
	# resolutions = ["100kb"]
	# gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)

	# target = "K562"
	# models = ["gru"]
	# run = ["run_2021-05-28-12-08-00"]
	# tags = ["gru (all chromosomes)"]
	# resolutions = ["100kb"]
	# gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)

	# target = "NHEK"
	# models = ["gru"]
	# run = ["run_2021-05-27-23-30-01"]
	# tags = ["gru (all chromosomes)"]
	# resolutions = ["100kb"]
	# gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)


	# target = "HMEC"
	# models = ["gru"]
	# run = ["run_2021-05-28-08-25-27"]
	# tags = ["gru (all chromosomes)"]
	# resolutions = ["100kb"]
	# gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)


	# target = "HUVEC"
	# models = ["gru"]
	# run = ["run_2021-05-28-20-15-10"]
	# tags = ["gru (all chromosomes)"]
	# resolutions = ["100kb"]
	# gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)

	# target = "IMR90"
	# models = ["gru"]
	# run = ["run_2021-05-27-22-38-47"]
	# tags = ["gru (all chromosomes)"]
	# resolutions = ["100kb"]
	# gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)

	##=========================================
	## 100 kb all chromosome gru + cross validation - worst model
	##=========================================

	target = "GM12878"
	models = ["gru"]
	run = ["run_2021-05-27-16-27-22"]
	tags = ["gru (all chromosomes)"]
	resolutions = ["100kb"]
	gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)

	target = "K562"
	models = ["gru"]
	run = ["run_2021-05-28-13-34-25"]
	tags = ["gru (all chromosomes)"]
	resolutions = ["100kb"]
	gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)

	target = "NHEK"
	models = ["gru"]
	run = ["run_2021-05-28-02-52-05"]
	tags = ["gru (all chromosomes)"]
	resolutions = ["100kb"]
	gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)


	target = "HMEC"
	models = ["gru"]
	run = ["run_2021-05-28-06-17-00"]
	tags = ["gru (all chromosomes)"]
	resolutions = ["100kb"]
	gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)


	target = "HUVEC"
	models = ["gru"]
	run = ["run_2021-05-28-19-17-36"]
	tags = ["gru (all chromosomes)"]
	resolutions = ["100kb"]
	gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)

	target = "IMR90"
	models = ["gru"]
	run = ["run_2021-05-27-21-12-43"]
	tags = ["gru (all chromosomes)"]
	resolutions = ["100kb"]
	gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)

	##=========================================
	## 100 kb all chromosome gru + cross validation (leave out NHEK) - all_chromosome_v4
	##=========================================
	# target = "IMR90"
	# models = ["gru"]
	# run = ["run_2021-05-27-21-12-43"]
	# tags = ["gru (all chromosomes)"]
	# resolutions = ["100kb"]
	# gnerate_curve(target,models,run,tags,resolutions,do_test,exp_dpath,args)












	



def gnerate_curve(target_cell,models,run,tags,resolutions,do_test,exp_dpath,args):
	
	test_results = {}

	for idx, (model_type, run) in enumerate(zip(models, run)):
		resolution = resolutions[idx]

		if args.split == "cross_validation":
			test_loader = torch.load(os.path.join("data","6_cell_input_updated",\
			"6_cell_input_updated_{}".format(args.resolution),"processed",\
			args.split,target_cell,"5_fold","test.pth"))
		else:
			test_loader = torch.load(os.path.join("data","6_cell_input_updated",\
			"6_cell_input_updated_{}".format(args.resolution),"processed",\
			args.split,target_cell,"test.pth"))

			validate_loader = torch.load(os.path.join("data","6_cell_input_updated",\
			"6_cell_input_updated_{}".format(args.resolution),"processed",\
			args.split,target_cell,"valid.pth"))
		
		print(os.path.join("data","6_cell_input_updated",\
			"6_cell_input_updated_{}".format(args.resolution),"processed",\
			args.split,target_cell,"5_fold","test.pth"))

		print(model_type)
		load_path = os.path.join("data","exp_data","cla","6_cell_input_updated_{}"\
			.format(resolution)\
			,model_type,run)
		print(load_path)
		model_param_path = os.path.join(load_path,"model_param.json")
		model_param = {}
		print()
		if os.path.isfile(model_param_path):
			with open(model_param_path, 'r') as j:
				model_param = json.loads(j.read())
		print(model_param)
		#load model
		if model_type == "lstm":
			model = lstm(100,model_param).to(device)
		elif model_type == "gru":
			model = gru(100,model_param).to(device)
		elif model_type[:3] == "cnn":
			n = model_type.split("_")[1]
			model = cnnNlayer(100,model_param, "cla").to(device)

		model.load_state_dict(torch.load(os.path.join(load_path,'model.pt')))
		
		if do_test ==True:
			print("Do test")
			results = test(model,test_loader,"cla")
		else:
			results = test(model,validate_loader,"cla")

		print(results["auroc"]["auroc_score"])

		tag = tags[idx]
		test_results[tag] = results


	for resolution in resolutions:
		print(resolution)
		mean_evec_path = os.path.join("data","mean_evec")
		if args.split == "cross_validation":
			mean_evec_folder = "mean_evec_updated_{}_allchromosome".format(resolution)
		else:
			mean_evec_folder = "mean_evec_updated_{}".format(resolution)
		


		if args.baseline == True:
			if do_test:
				predict_data_path = os.path.join(mean_evec_path,mean_evec_folder,"{}_test.txt".format(target_cell))
			else:
				predict_data_path = os.path.join(mean_evec_path,mean_evec_folder,"{}_validate.txt".format(target_cell))
			
			f = open(predict_data_path, "r")
			predicts = f.readlines()

			all_lines = []
			# target_data_dir = os.path.join("data","Compartments",target_cell,"100kb")


			target_data_dir = os.path.join("data","updated_eigenvectors_{}".format(resolution),target_cell)
			if args.split == "5_cells":
				if do_test:
					chr_range = (19,23)
				else:
					chr_range = (17,19)
			elif args.split =="cross_validation":
				if do_test:
					chr_range = (1,23)

			for idx in range(chr_range[0],chr_range[1]):
				file_path = os.path.join(target_data_dir,"{}_eigen_{}_chr{}.csv".format(target_cell,resolution,idx))
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

			data = {}
			data["auroc"] = {}
			data["auroc"]["auroc_score"] = auroc
			data["auroc"]["fpr"] = fpr
			data["auroc"]["tpr"] = tpr

			data["auprc"] = {}
			data["auprc"]["auprc_score"] = auprc
			data["auprc"]["recall"] = recall
			data["auprc"]["precision"] = precision

			test_results["mean_{}".format(resolution)] = data

	if target_cell != None:
		combine_plot_curve_with_target(test_results,exp_dpath,target_cell,do_test)
	else:
		combine_plot_curve(test_results,exp_dpath)




if __name__ == "__main__":
    main()