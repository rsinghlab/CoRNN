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

from preprocess import *
from model import *
from utils import *
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)



def train(model, train_loader, valid_loader, loss_fn, args):
	print("===Training...")
	model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
	#TODO: add learning rate decay: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

	max_auroc = 0
	best_epoch_auroc = 0
	max_auprc = 0
	best_epoch_auprc = 0
	best_model = {}

	for epoch in range(args.epoch):
		print("Epoch : {}".format(epoch))
		for param_group in optimizer.param_groups:
			print("learning rate:",param_group['lr'])
		for batch in tqdm(train_loader):

			# print(batch.keys())
			model.train()
			inputs = batch["input"]

			if args.task == "cla":
				labels = batch["cla_labels"]
			elif args.task == "reg":
				labels = batch["reg_labels"]
			inputs = inputs.to(device)
			labels = labels.to(device)

			mean_evec = None
			if args.add_mean_evec:
				mean_evec = batch["mean_evec"]
				mean_evec = mean_evec.to(device)

			optimizer.zero_grad()
			y_pred = model.forward(inputs,mean_evec)

			if args.task == "reg":
				loss = loss_fn(y_pred.flatten(), labels.flatten())
			elif args.task == "cla":
				loss = loss_fn(y_pred, labels)
			#print(loss)
			loss.backward()
			optimizer.step()


		valid_loss, auroc, auprc, auroc_data, auprc_data = valid(model, valid_loader, loss_fn, args, max_auroc,max_auprc, epoch)
		#valid_loss = 1-auroc
		if args.scheduler == True:
				scheduler.step(valid_loss)


		if args.task == "cla":
			if auroc > max_auroc:
				best_epoch_auroc = epoch
				max_auroc = auroc
				best_auroc_data = auroc_data
				#save model
				torch.save(model.state_dict(), os.path.join(args.exp_dpath,'model.pt'))

				if args.save:
					print("save model!!")
					best_model = model.state_dict()
					
			if auprc > max_auprc:
				best_epoch_auprc = epoch
				max_auprc = auprc
				best_auprc_data = auprc_data

	if args.task == "cla":
		#log the epoch that generate the best validation AUROC
		experiment.log_metric("Epoch with best AUROC", best_epoch_auroc)
		experiment.log_metric("Best AUROC: ",max_auroc)
		
		experiment.log_metric("Epoch with best AUPRC", best_epoch_auprc)
		experiment.log_metric("Best AUPRC: ",max_auprc)
		#log the AUROC graph with best AUROC score
		image_output_dir = os.path.join(args.exp_dpath,"AUROC_img")
		if not os.path.exists(image_output_dir):
			os.makedirs(image_output_dir)	
		experiment.log_image(os.path.join(image_output_dir,"AUROC_epoch_{}.png".format(best_epoch_auroc)))

		image_output_dir = os.path.join(args.exp_dpath,"AUPRC_img")
		if not os.path.exists(image_output_dir):
			os.makedirs(image_output_dir)	
		experiment.log_image(os.path.join(image_output_dir,"AUPRC_epoch_{}.png".format(best_epoch_auprc)))

		
		#save best auroc and best auprc data
		# save_dpath = os.path.join("data","exp_data")
		# if not os.path.exists(save_dpath):
		# 	os.makedirs(save_dpath)

		save_file = os.path.join(args.exp_dpath,"curve_info.json")
		save_data = {}
		if os.path.isfile(save_file):
			with open(save_file, 'r') as j:
				save_data = json.loads(j.read())

		save_data[args.split] = {"auroc_score":max_auroc,"auprc_score":max_auprc, "auprc":best_auprc_data,"auroc":best_auroc_data}

		with open(save_file, 'w') as outfile:
			json.dump(save_data, outfile)
		
		# auprc_data_dir = os.path.join(args.exp_dpath,"auprc.json")
		# with open(auprc_data_dir, 'w') as outfile:
		# 	json.dump(best_auprc_data, outfile)

		# experiment.log_metric("auprc_data",best_auprc_data)
		# experiment.log_metric("auroc_data",best_auroc_data)
		return best_model, max_auroc

def train_cross_validation(model,loss_fn,args):

	print("===Training...")
	model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
	#TODO: add learning rate decay: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

	max_auroc = 0
	best_fold = 0
	
	best_model = {}

	# test_loader = torch.load(os.path.join(args.data_save_dir,"test.pth"))

	for fold in range(args.num_fold):
		print("***Fold {}***".format(fold))
		fold_data_path = os.path.join(args.data_save_dir, str(fold))
		train_loader = torch.load(os.path.join(fold_data_path,"train.pth"))
		valid_loader = torch.load(os.path.join(fold_data_path,"valid.pth"))
		args.current_fold = fold
		fold_best_model, auroc = train(model, train_loader, valid_loader, loss_fn, args)

		if auroc > max_auroc:
			best_model	 = fold_best_model
			max_auroc = auroc
			best_fold = fold
		
		# auprc_data_dir = os.path.join(args.exp_dpath,"auprc.json")
		# with open(auprc_data_dir, 'w') as outfile:
		# 	json.dump(best_auprc_data, outfile)

		# experiment.log_metric("auprc_data",best_auprc_data)
		# experiment.log_metric("auroc_data",best_auroc_data)
		return best_model


def valid_cross_validation(model,loss_fn,args):

	print("===Training...")
	model.eval()
	# optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)
	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
	#TODO: add learning rate decay: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

	max_auroc = 0
	best_fold = 0
	
	best_model = {}

	# test_loader = torch.load(os.path.join(args.data_save_dir,"test.pth"))

	for fold in range(args.num_fold):
		print("***Fold {}***".format(fold))
		fold_data_path = os.path.join(args.data_save_dir, str(fold))
		# train_loader = torch.load(os.path.join(fold_data_path,"train.pth"))
		valid_loader = torch.load(os.path.join(fold_data_path,"valid.pth"))
		args.current_fold = fold

		valid(model, valid_loader, loss_fn, args, 0, 0, 0)
		
		# fold_best_model, auroc = train(model, train_loader, valid_loader, loss_fn, args)

		# if auroc > max_auroc:
		# 	best_model	 = fold_best_model
		# 	max_auroc = auroc
		# 	best_fold = fold
		
		# auprc_data_dir = os.path.join(args.exp_dpath,"auprc.json")
		# with open(auprc_data_dir, 'w') as outfile:
		# 	json.dump(best_auprc_data, outfile)

		# experiment.log_metric("auprc_data",best_auprc_data)
		# experiment.log_metric("auroc_data",best_auroc_data)
		return best_model

def valid(model, valid_loader, loss_fn, args, max_auroc,max_auprc, epoch):
	model.eval()
	# loss_fn = nn.CrossEntropyLoss()
	cur_loss = []
	cur_pred = []
	cur_labels = []
	for batch in tqdm(valid_loader):
		inputs = batch["input"]
		inputs = inputs.to(device)
		if args.task == "cla":
			labels = batch["cla_labels"].to(device)
		elif args.task == "reg":
			labels = batch["reg_labels"].to(device)

		mean_evec = None
		if args.add_mean_evec:
			mean_evec = batch["mean_evec"]
			mean_evec = mean_evec.to(device)
		
		y_pred = model.forward(inputs,mean_evec)
		
		if args.task == "reg":
			loss = loss_fn(y_pred.flatten(), labels.flatten())
		elif args.task == "cla":
			loss = loss_fn(y_pred, labels)
		loss = loss.cpu().detach().numpy()
		cur_loss.append(loss)
		# print(type(cur_pred))
		# print(type(y_pred[:,-1].detach()))
		# print(y_pred)
		cur_pred += y_pred[:,-1].detach().tolist()
		cur_labels += labels.detach().tolist()

	mean_loss = np.mean(cur_loss)
	experiment.log_metric("Validation Loss", mean_loss)
	



	if args.task == "cla":
		# Calculate validation AUROC after each training epoch, save the generate graph to ./AUROC_img/
		
		if args.cross_validation == True:
			img_save_dir = os.path.join(args.exp_dpath,"fold_{}".format(args.current_fold))
			if not os.path.exists(img_save_dir):
				os.makedirs(img_save_dir) 
		else:
			img_save_dir = args.exp_dpath
		
		AUROC_image_output_dir = os.path.join(img_save_dir,"AUROC_img")
		if not os.path.exists(AUROC_image_output_dir):
			os.makedirs(AUROC_image_output_dir)

		AUPRC_image_output_dir = os.path.join(img_save_dir,"AUPRC_img")
		if not os.path.exists(AUPRC_image_output_dir):
			os.makedirs(AUPRC_image_output_dir)
		
		AUROC_image_output_path = os.path.join(AUROC_image_output_dir,"AUROC_epoch_{}.png".format(epoch))
		AUPRC_image_output_path = os.path.join(AUPRC_image_output_dir,"AUPRC_epoch_{}.png".format(epoch))
		#curve_name = "AUROC"
		# print("cur_labels",cur_labels)
		# print("cur_pred",cur_pred)
		# for idx, label in enumerate(cur_labels):
		# 	try label == "NaN":
		# 		print(idx, label)
		# 	except:
		# 		continue
		# for idx, label in enumerate(cur_pred):
		# 	try label.lower() == "nan":
		# 		print(idx, label)
		# 	except:
		# 		continue



		auroc, fpr, tpr = AUROC(experiment,cur_labels,cur_pred,AUROC_image_output_path, "AUROC", save_fig = True)
		auprc,recall, precision = AUPRC(experiment,cur_labels,cur_pred,AUPRC_image_output_path, "AUPRC", save_fig = True)
		# f1 = F1(experiment, cur_labels, cur_pred)
		# log the AUROC value
		experiment.log_metric("Validation AUROC",auroc)
		experiment.log_metric("Validation AUPRC",auprc)
		# experiment.log_metric("Validation f1",f1)
		
		return mean_loss,auroc, auprc, {"fpr":fpr.tolist(), "tpr":tpr.tolist()}, {"recall":recall.tolist(),"precision":precision.tolist()}

	elif args.task == "reg":
		evaluate_regression(experiment, cur_labels, cur_pred)
		return None, None


def test(model, test_loader, loss_fn,args):
	print("===Testing...")
	model.eval()
	loss_fn = nn.CrossEntropyLoss()
	cur_loss = []
	cur_pred = []
	cur_labels = []
	for batch in tqdm(test_loader):
		inputs = batch["input"]
		inputs = inputs.to(device)
		if args.task == "cla":
			labels = batch["cla_labels"]
		elif args.task == "reg":
			labels = batch["reg_labels"]
		inputs = inputs.to(device)
		labels = labels.to(device)

		if args.add_mean_evec:
			mean_evec = batch["mean_evec"]
			mean_evec = mean_evec.to(device)
		
		y_pred = model.forward(inputs,mean_evec)

		loss = loss_fn(y_pred, labels)
		loss = loss.cpu().detach().numpy()
		cur_loss.append(loss)
		cur_pred += y_pred[:,-1].detach().tolist()
		cur_labels += labels.detach().tolist()
	
	mean_loss = np.mean(cur_loss)
	
	image_output_dir = os.path.join(args.exp_dpath,"AUROC_img")
	if not os.path.exists(image_output_dir):
		os.makedirs(image_output_dir)

	image_output_path = os.path.join(image_output_dir,"testing_AUROC.png")
	curve_name = "AUROC"
	auroc, fpr, tpr = AUROC(experiment,cur_labels,cur_pred,image_output_path, curve_name,save_fig = True)
	
	image_output_dir = os.path.join(args.exp_dpath,"AUPRC_img")
	if not os.path.exists(image_output_dir):
		os.makedirs(image_output_dir)

	image_output_path = os.path.join(image_output_dir,"testing_AUROC.png")
	curve_name = "AUPRC"
	auprc, recall, precision = AUPRC(experiment,cur_labels,cur_pred,image_output_path, curve_name,save_fig = True)
	
	experiment.log_image(image_output_path)
	experiment.log_metric("Testing Loss", mean_loss)
	experiment.log_metric("Testing AUROC",auroc)
	experiment.log_metric("Testing AUPRC",auprc)





def load_and_process_data(args):
	#read all data file, use chromosome as key 

	if args.split != None:
		data_save_dir = os.path.join(args.data_dir,"processed",args.split,args.cell)
		print(data_save_dir)
		if not os.path.exists(data_save_dir):
			os.makedirs(data_save_dir)	
	else:
		data_save_dir = os.path.join(args.data_dir,"processed",args.split)
		print(data_save_dir)
		if not os.path.exists(data_save_dir):
			os.makedirs(data_save_dir)	
	

	if os.path.exists(os.path.join(data_save_dir,"train.pth")) and \
		os.path.exists(os.path.join(data_save_dir,"valid.pth")) and \
			os.path.exists(os.path.join(data_save_dir,"test.pth")):
		print("***Loading cached .pth files...")
			# train_loader = abDataset()
		train_loader = torch.load(os.path.join(data_save_dir,"train.pth"))
		valid_loader = torch.load(os.path.join(data_save_dir,"valid.pth"))
		test_loader = torch.load(os.path.join(data_save_dir,"test.pth"))
		print("Finish loading.")	
		return train_loader,valid_loader,test_loader

	print("***Loading raw .cvs files...")
	all_data = {}
	delimiters = ".", "_"
	
	if args.split == "5_cells":
		training_data, validation_data, testing_data = \
						prepare_data_cell_lines_v6(args,args.data_dir,args.cell,args.use_corre)

	elif args.split == "single":
		training_data, validation_data, testing_data = \
						prepare_data_cell_lines_single(args,args.data_dir,args.cell,args.use_corre)
	elif args.split == "all_chr_temp":
		training_data, validation_data, testing_data = \
						prepare_data_cell_lines_v6(args,args.data_dir,args.cell,args.use_corre)

	else:
		for filename in os.listdir(args.data_dir):
			if filename.endswith(".csv"): 
				print("***Reading data from {}...".format(filename))
				chromosome = filename.split(".")[0]
				filepath = os.path.join(args.data_dir,filename)
				with open(filepath, newline='') as csvfile:
					data_reader = csv.reader(csvfile, delimiter=',')
					lines = []
					for row in data_reader:
						lines.append(row)
					all_data[chromosome] = lines


	print("Finish loading.")
	cleaned_data = {}

	for key, item in all_data.items():
		cleaned_data[key] = clean_data(item)

	print("Finish Cleaning.")


	if args.split == "random":
		#random split of data
		print("Split data by random.")

		combined_data = combine_data(cleaned_data)

		postive_data = sum([item["cla_label"] for item in combined_data])
		negative_data = len(combined_data) - postive_data
		max_label = max([item["reg_label"] for item in combined_data])
		min_label = min([item["reg_label"] for item in combined_data])

		# print("===Total number of data: {}".format(len(combined_data)))
		# print("===Number of data with label 1: {}".format(postive_data))
		# print("===Number of data with label 0: {}".format(negative_data))
		# print("===Max label value: {}".format(max_label))
		# print("===min label value: {}".format(min_label))

		training_data, validation_data, testing_data = random_split_data(combined_data,0.7,0.15)

	if args.split == "imr90":

		#split data by chromosome
		#chr1-chr16 for training, 17-18 for validation, 19-22 for evaludation
		print("Split data by chromosome.")

		train_dict = {}
		valid_dict = {}
		test_dict = {}

		for i in range(1,17):
			train_dict[i] = cleaned_data["IMR90_chr{}".format(str(i))]
		for i in range(17,19):
			valid_dict[i] = cleaned_data["IMR90_chr{}".format(str(i))]
		for i in range(19,23):
			test_dict[i] = cleaned_data["IMR90_chr{}".format(str(i))]
 
		print("Finish spliting.")
		#print(train_dict)

		training_data = combine_data(train_dict)
		validation_data = combine_data(valid_dict)
		testing_data = combine_data(test_dict)


	print("=====Training data info=====")
	# print(training_data)
	postive_data = sum([item["cla_label"] for item in training_data])
	negative_data = len(training_data) - postive_data
	max_label = max([item["reg_label"] for item in training_data])
	min_label = min([item["reg_label"] for item in training_data])

	print("Total number of training data: {}".format(len(training_data)))
	print("Number of data with label 1: {}".format(postive_data))
	print("Number of data with label 0: {}".format(negative_data))
	print("Max label value: {}".format(max_label))
	print("min label value: {}".format(min_label))

	print("=====Validation data info=====")

	postive_data = sum([item["cla_label"] for item in validation_data])
	negative_data = len(validation_data) - postive_data
	max_label = max([item["reg_label"] for item in validation_data])
	min_label = min([item["reg_label"] for item in validation_data])

	print("Total number of training data: {}".format(len(validation_data)))
	print("Number of data with label 1: {}".format(postive_data))
	print("Number of data with label 0: {}".format(negative_data))
	print("Max label value: {}".format(max_label))
	print("min label value: {}".format(min_label))

	print("=====Testing data info=====")

	postive_data = sum([item["cla_label"] for item in testing_data])
	negative_data = len(testing_data) - postive_data
	max_label = max([item["reg_label"] for item in testing_data])
	min_label = min([item["reg_label"] for item in testing_data])

	print("Total number of training data: {}".format(len(testing_data)))
	print("Number of data with label 1: {}".format(postive_data))
	print("Number of data with label 0: {}".format(negative_data))
	print("Max label value: {}".format(max_label))
	print("min label value: {}".format(min_label))



	#create dataset
	train_dataset = abDataset(training_data)
	valid_dataset = abDataset(validation_data)
	test_dataset = abDataset(testing_data)

	# create DataLoader
	train_loader = DataLoader(train_dataset,batch_size=args.train_batch_size, shuffle=True)
	valid_loader = DataLoader(valid_dataset,batch_size=args.valid_batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False)

	#save dataloader
	if not os.path.exists(data_save_dir):
		os.makedirs(data_save_dir)	
	torch.save(train_loader,os.path.join(data_save_dir,"train.pth"))
	torch.save(valid_loader,os.path.join(data_save_dir,"valid.pth"))
	torch.save(test_loader,os.path.join(data_save_dir,"test.pth"))

	return train_loader,valid_loader,test_loader 



def load_and_process_cross_validation_data(args):

	# check if datasets exist or not
	cell_lines = ["IMR90", "HMEC", "K562", "GM12878","HUVEC"]

	data_processed = True
	datasets = {}

	if len(cell_lines) < 6:
		folder_name = '-'.join(cell_lines)
		data_save_dir = os.path.join(args.data_dir,"processed",\
		"cross_validation",\
		"{}_cells".format(len(cell_lines)),
		folder_name,
		args.cell,\
		"{}_fold".format(args.num_fold))
	else:
		data_save_dir = os.path.join(args.data_dir,"processed",\
		"cross_validation",\
		args.cell,\
		"{}_fold".format(args.num_fold))

	args.data_save_dir = data_save_dir
	
	print(args.data_save_dir)
	if not os.path.exists(args.data_save_dir):
		os.makedirs(args.data_save_dir)

	#if test loader exist, load it
	if os.path.exists(os.path.join(args.data_save_dir,"test.pth")):
		test_loader = torch.load(os.path.join(data_save_dir,"test.pth"))
		#datasets["test"] = test_loader
		print("test files exist.")
	else:
		data_processed = False



	# if 
	for fold_id in range(args.num_fold):
		fold_save_path = os.path.join(args.data_save_dir, str(fold_id))
		if not os.path.exists(fold_save_path):
			os.makedirs(fold_save_path)
		# print(fold_save_path)
		if os.path.exists(os.path.join(fold_save_path,"train.pth")) and \
		os.path.exists(os.path.join(fold_save_path,"valid.pth")):
			print("fold {} .pth train and validare files exist.".format(fold_id))
			# print("***Loading cached fold {} .pth files...".format(fold_id))
			# train_loader = torch.load(os.path.join(fold_save_path,"train.pth"))
			# valid_loader = torch.load(os.path.join(fold_save_path,"valid.pth"))
			# test_loader = torch.load(os.path.join(fold_save_path,"test.pth"))
			# print("Finish loading.")
			# datasets[fold_id]["train"] = train_loader 
			# datasets[fold_id]["valid"] = valid_loader
		else:
			data_processed = False

	if data_processed == False:
		testing_data = prepare_data_cross_validation(args, cell_lines)
		test_dataset = abDataset(testing_data)
		test_loader = DataLoader(test_dataset,batch_size=args.train_batch_size, shuffle=True)
	

	return test_loader	

def load_and_process_cross_validation_data_with_mean_evec(args):

	# check if datasets exist or not
	cell_lines = ["IMR90", "HMEC", "K562", "GM12878","HUVEC","NHEK"]

	data_processed = True
	datasets = {}

	if len(cell_lines) < 6:
		folder_name = '-'.join(cell_lines)
		data_save_dir = os.path.join(args.data_dir,"processed",\
		"cross_validation_with_mean_debug",\
		"{}_cells".format(len(cell_lines)),
		folder_name,
		args.cell,\
		"{}_fold".format(args.num_fold))
	else:
		data_save_dir = os.path.join(args.data_dir,"processed",\
		"cross_validation_with_mean_debug",\
		args.cell,\
		"{}_fold".format(args.num_fold))

	args.data_save_dir = data_save_dir
	
	print(args.data_save_dir)
	if not os.path.exists(args.data_save_dir):
		os.makedirs(args.data_save_dir)

	#if test loader exist, load it
	if os.path.exists(os.path.join(args.data_save_dir,"test.pth")):
		test_loader = torch.load(os.path.join(data_save_dir,"test.pth"))
		#datasets["test"] = test_loader
		print("test files exist.")
	else:
		data_processed = False


	for fold_id in range(args.num_fold):
		fold_save_path = os.path.join(args.data_save_dir, str(fold_id))
		if not os.path.exists(fold_save_path):
			os.makedirs(fold_save_path)
		# print(fold_save_path)
		if os.path.exists(os.path.join(fold_save_path,"train.pth")) and \
		os.path.exists(os.path.join(fold_save_path,"valid.pth")):
			print("fold {} .pth train and validare files exist.".format(fold_id))

		else:
			data_processed = False

	if data_processed == False:
		testing_data = prepare_data_cross_validation_with_mean_evec_debug(args, cell_lines)
		test_dataset = abDataset(testing_data)
		test_loader = DataLoader(test_dataset,batch_size=args.train_batch_size, shuffle=True)
	

	return test_loader

def load_and_process_testing_by_region(args):
	cell_lines = ["IMR90", "HMEC", "K562", "GM12878","HUVEC","NHEK"]

	prepare_test_data_by_region(args,cell_lines)

def main():
    parser = argparse.ArgumentParser()

    #required argument
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The directory contains all input files/or cell line folders",
    )
    parser.add_argument(
        "--model",
        default="cnn_1",
        type=str,
        required=True,
        help="The directory contains all input files, all files should end with .cvs for this task",
    )

    parser.add_argument(
        "--task",
        default="cla",
        type=str,
        required=True,
        help="Classification or Regression, cla or reg",
    )
    

    #other argument
    parser.add_argument("-l", "--load", default = None, type = str,
                        help="load model.pt and model param")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-v", "--valid", action="store_true",
                        help="run validation loop")
    
    parser.add_argument(
        "--load_dataset",
        default=False,
        type=bool,
        help="if all data has been processed before and have dataset saved in the data_dir, load the dataset directly",
    )

    #training setting
    parser.add_argument(
        "--split", default="random", type=str, help="partition approach, 'random’ or 'imr90' or 'cell_lines",
    )
    parser.add_argument(
        "--train_batch_size", default=64, type=int, help="Batch size for training",
    )
    parser.add_argument(
        "--valid_batch_size", default=1, type=int, help="Batch size for validation",
    )
    parser.add_argument(
        "--test_batch_size", default=1, type=int, help="Batch size for testing",
    )
    parser.add_argument(
    	"--epoch",default = 20, type=int, help="number of training epoch",
    )
    parser.add_argument(

    	"--learning_rate",default = 0.001, type=float, help="The initial learning rate for Adam."
    )

    parser.add_argument(
    	"--weight_decay", default = 0, type=float, help = "L2 regularization"
    )

    #fixed model parameters
    parser.add_argument(
        "--seq_len", default=100, type=int, help="Do not change. sequence length of the input data, should be 100 in this task",
    )

    parser.add_argument(
        "--exp_dpath", default=None, type=str, help="Path for saving the current experiment run",
    )

    parser.add_argument(
        "--data_save_dir", default=None, type=str, help="Path for saving the processed .pth data files",
    )

    parser.add_argument(
    	"--cell", default=None, type=str, help="the cell line for validation and testing",
    )

    parser.add_argument(
    	"--use_corre", default=False, type=bool, help="True when need correlation value for flipping the sign of eigenvectors",
    )
    
    parser.add_argument(
    	"--config", default=None, type=str, help="path to config file"
    )

    parser.add_argument(
    	"--resolution", default="50kb", type=str, help="experiment resolution of eigenvectors"
    )

    parser.add_argument(
    	"--scheduler", default=False, type=bool, help="Use scheduler for learning rate"
    )

    parser.add_argument(
    	"--hidden", default=32, type=int, help="number of hidden states for rnn"
    )

    parser.add_argument(
    	"--layer", default=1, type=int, help="number of layer in rnn"
    )

    parser.add_argument(
    	"--cross_validation", default=False, type=bool, help="use cross validation or not"
    )

    parser.add_argument(
    	"--num_fold", default=None, type=int, help="number of fold in cross validation"
    )

    parser.add_argument(
    	"--current_fold", default=0, type=int, help="the current number of fold"
    )

    parser.add_argument(
    	"--add_mean_evec", default=False, type=bool, help="use cross validation or not"
    )

    parser.add_argument(
    	"--special_tag", default=None, type=str, help="number of fold in cross validation"
    )

    parser.add_argument(
    	"--test_by_region", default=None, type=bool, help="number of fold in cross validation"
    )
    args = parser.parse_args()
    
    if args.load:
    	saved_hyper_params = {}
    	params_save_path = os.path.join("data","exp_data",args.task,args.data_dir.split("/")[1],args.model,args.load,"hyperparameters.json")
    	with open(params_save_path, 'r') as j:
    			saved_hyper_params = json.loads(j.read())

    	args.train_batch_size = saved_hyper_params["train_batch_size"]
    	args.test_batch_size = saved_hyper_params["test_batch_size"]
    	args.valid_batch_size = saved_hyper_params["valid_batch_size"]
    	args.learning_rate = saved_hyper_params["learning_rate"]

    #Todo: load hyper parameters

    if not os.path.isdir(os.path.join("data","exp_data")):
    	os.mkdir(os.path.join("data","exp_data"))
    if not os.path.isdir(os.path.join("data","exp_data",args.task)):
    	os.mkdir(os.path.join("data","exp_data",args.task))	

    if not os.path.isdir(os.path.join("data","exp_data",args.task,args.data_dir.split("/")[-1])):
    	os.mkdir(os.path.join("data","exp_data",args.task,args.data_dir.split("/")[-1]))



    exp_dpath = os.path.join("data","exp_data",args.task,args.data_dir.split("/")[-2],args.model)
    if not os.path.isdir(exp_dpath):														
    	os.mkdir(exp_dpath)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    print("hi there",exp_dpath)
    

    run_dpath = os.path.join(exp_dpath, "run_{}".format(now_str))
    if not os.path.isdir(run_dpath):
    	os.mkdir(run_dpath)
    args.exp_dpath = run_dpath

    if args.test_by_region:
    	load_and_process_testing_by_region(args)
    	exit(0)

    if args.cross_validation:
    	if args.add_mean_evec:
    		test_loader = load_and_process_cross_validation_data_with_mean_evec(args)
    	else:test_loader = load_and_process_cross_validation_data(args)
    else:
    	train_loader, valid_loader, test_loader = load_and_process_data(args)
	   

    # load model parameters
    if args.load:
    	#read parameters from trained model folder
    	load_path = os.path.join("data","exp_data",args.task,args.data_dir.split("/")[1],args.model,args.load)
    	model_param_path = os.path.join(load_path,"model_param.json")
    	model_param = {}
    	if os.path.isfile(model_param_path):
    		with open(model_param_path, 'r') as j:
    			model_param = json.loads(j.read())
    if args.model == "lstm":
   		model_param = {
    		"hidden_dim": 256,
    		"n_layers": 1
    		}
    elif args.model == "gru":
    	model_param = {
    		"hidden_dim": args.hidden,
    		"n_layers": args.layer,
    		"bidirectional": False,
    		"mean_evec":args.add_mean_evec
    		}

    elif args.model[:3] == "cnn":
    	#read parameters from confid file
    	#Suzy TODO: write a seperate file for lstm config
    	#Suzy TODO: rewrite the config loading of cnn
    	model_config_dpath = os.path.join("data","model_config_1.json")
    	if args.config: # if args.config != None
    		model_config_dpath = args.config

    	with open(model_config_dpath) as f:
    		model_config = json.load(f)
    	n = args.model.split("_")[1]
    	model_param = model_config["cnn{}_param".format(n)]
    	#save the parameters to the exp_run folder
    
    # if args.hidden:
    # 	model_param["hidden_dim"] = args.hidden
    # if args.layer:
    # 	model_param["n_layers"] = args.layer
    save_param_file = os.path.join(args.exp_dpath,"model_param.json")
    with open(save_param_file, 'w') as outfile:
    	json.dump(model_param, outfile)

    if args.model == "lstm":
    	print(model_param)
    	model = lstm(args.seq_len,model_param).to(device)
    elif args.model == "gru":
    	print(model_param)
    	model = gru_debug(args.seq_len,model_param).to(device)
    elif args.model[:3] == "cnn":
    	n = args.model.split("_")[1]
    	model = cnnNlayer(args.seq_len,model_param, args.task).to(device)

    

    #Print model and task info	
    print("======Experiment information:======")
    print("Task:  				{}".format("regression" if args.task == "reg" else "classification"))
    print("Target cell:			{}".format(args.cell))
    print("Data Dir:   			{}".format(args.data_dir))
    print("Output Dir: 			{}".format(args.exp_dpath))
    print("Model: 				{}".format(args.model))
    print("Training epoch : 		{}".format(args.epoch))
    print("Learning rate : 		{}".format(args.learning_rate))
    print("Model parameters : 	{}".format(model_param))

    # experiment.add_tag(args.task)
    # experiment.add_tag("updated")
    experiment.add_tag(args.model)
    experiment.add_tag(args.split)
    experiment.add_tag(args.resolution)
    if args.cell != None:
    	experiment.add_tag(args.cell)
    experiment.log_text(args.exp_dpath)

    if args.cross_validation == True:
    	experiment.add_tag("cross_validation")

    if args.special_tag != None:
    	experiment.add_tag(args.special_tag)

    # model = cnn3layer(args.seq_len,args.cnn3).to(device)
    if args.task == "cla":
    	loss_fn = nn.CrossEntropyLoss()
    elif args.task == "reg":
    	loss_fn = nn.L1Loss() # Mean absolute error
    	# loss_fn = nn.MSELoss()  # Mean squares error

    if args.load:
    	model.load_state_dict(torch.load(os.path.join(load_path,'model.pt')))
    if args.train:
    	if args.cross_validation:
    		best_model = train_cross_validation(model,loss_fn,args)
    	else:
    		best_model,_ = train(model, train_loader, valid_loader, loss_fn, args)
    
    if args.valid:
    	if args.cross_validation:
    		valid_cross_validation(model,loss_fn,args)
    	else:
    		valid(model,loss_fn,args,0,0,0)
    if args.test:
    	test(model, test_loader,loss_fn, args)
    if args.save:
    	print("===Saving model.pt")
    	print("===Saving to {}".format(args.exp_dpath))
    	torch.save(best_model, os.path.join(args.exp_dpath,'model.pt'))
    	
    	# The model is being saved during the training process, 
    	# this section is to save the hyperparameters

    	hyperparam = vars(args)
    	hp_dpath = os.path.join(args.exp_dpath,"hyperparams.json")
    	with open(hp_dpath, 'w') as outfile:
    		json.dump(hyperparam, outfile)



    print(now_str)
    print(args.exp_dpath)

if __name__ == "__main__":
    main()