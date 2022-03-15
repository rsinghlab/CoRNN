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

from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_binary(val):
	if float(val) >= 0.5:
		return 1
	else:
		return 0

def test(model,test_loader,task):
	# print("===Testing...")
	model.eval()
	loss_fn = nn.CrossEntropyLoss()
	cur_loss = []
	cur_pred = []
	cur_labels = []
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
		y_pred = model.forward(inputs,mean_evec)
		loss = loss_fn(y_pred, labels)
		loss = loss.cpu().detach().numpy()
		cur_loss.append(loss)
		cur_pred += y_pred[:,-1].detach().tolist()
		cur_labels += labels.detach().tolist()
	
	#calculate accuracy
	pred_binary = [to_binary(val) for val in cur_pred]

	
	return cur_pred,pred_binary

def load_model_and_run(model_path,test_loader):

	model_param_path = os.path.join(model_path,"model_param.json")

	model_param = {}
	print()
	if os.path.isfile(model_param_path):
		with open(model_param_path, 'r') as j:
			model_param = json.loads(j.read())
	print(model_param)

	#load model

	model = gru(100,model_param).to(device)
	
	model.load_state_dict(torch.load(os.path.join(model_path,'model.pt')))
	
	pred, pred_binary = test(model,test_loader,"cla")

	return pred, pred_binary

def prepare_data_cell_specific():
	
	cell_lines = ["IMR90", "NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	# cell_lines = ["IMR90", "HUVEC"]
	# read all data, store with key cellName_chrX: all data
	all_data = {}
	train_data_all = {}
	test_data_all = {}

	for cell_line in cell_lines:
		cell_line_dir = os.path.join("data/6_cell_input_updated/6_cell_input_updated_100kb/",cell_line)
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

	
	mean_evec_dir = os.path.join("data","mean_evec","mean_evec_updated_{}_allchromosome_by_chromosome".format("100kb"))
	region_dir = os.path.join("data","region","region_{}_by_chromosome".format("100kb"))
	updated_evec_dir = os.path.join("data","updated_eigenvectors_{}".format("100kb"))


	mean_evec = {}
	region_data = {}

	#process mean and region file based on the eigenvectors files

	print("***Load mean and regions***")
	
	for cell in cell_lines:
		for ch in range(1,23):
			# in this setting, the mean eigenvector is always the mean of 5 cells exclude the target cell

			print("{}: chr {}".format(cell,ch))

			#load region file
			region_file_path = os.path.join(region_dir,cell,"region_{}_chr{}.txt".format(cell,ch))
			region_f = open(region_file_path,"r")
			region_lines = region_f.readlines()
			region_lines = [float(line.strip().lower()) for line in region_lines]


			#load mean eigenvector
			load_path = os.path.join(mean_evec_dir,cell,"{}_chr{}_test.txt".format(cell,ch))
			f = open(load_path, "r")
			read_mean = f.readlines()
			all_mean = [mean.strip().lower() for mean in read_mean]

			#load eigenvectors
			updated_evec_file = os.path.join("data","updated_eigenvectors_{}".format("100kb"),cell,\
				"{}_eigen_{}_chr{}.csv".format(cell,"100kb",ch))
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
		print("cleaning ",key)
		cell = key.split("_")[0]
		chr_num = key.split("_")[1]
		
		corre = None
		
		curr_mean_evec = mean_evec[key]
		curr_region = region_data[key]
		# print(key)
		cleaned_data[key] = clean_data_for_cell_specific(item,corre,curr_mean_evec,curr_region)
	# exit()
	#divide data into train, validate, testing	

	test_chrom = {}
	print("finishi cleaning")
	
	save_dir = os.path.join("data","100kb_cell_specific_regions")
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	for cell in cell_lines:
		if not os.path.exists(os.path.join(save_dir,cell)):
			os.makedirs(os.path.join(save_dir,cell))


	for cell in cell_lines:
		test_chrom[cell] = {}


	for chrom, data in cleaned_data.items():
		# print(chrom)
		cell = chrom.split("_")[0]
		chr_num = chrom.split("_")[1][3:]
		test_chrom[cell][chrom] = data 


	for cell, data in test_chrom.items():
		combined_data = combine_data(data)

		test_dataset = abDataset_with_mean_and_region(combined_data)
		test_loader = DataLoader(test_dataset,batch_size=1, shuffle=False)
		data_save_dir = os.path.join(save_dir,cell)
		torch.save(test_loader,os.path.join(data_save_dir,"{}_test.pth".format(cell)))
		print("save file: ",os.path.join(data_save_dir,"{}_test.pth".format(cell)))


def main():

	cells = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]

	runs = {
	"IMR90": "run_2021-07-16-03-31-59",
	"K562":"run_2021-07-17-12-03-51", 
	"NHEK":"run_2021-07-13-08-55-26",
	 "HMEC":"run_2021-07-15-02-21-42", 
	 "GM12878":"run_2021-07-16-12-53-39", 
	 "HUVEC":"run_2021-07-14-20-41-53"
	}

	prepare_data_cell_specific()

	test_save_dir = os.path.join("data","100kb_cell_specific_regions","test_results")
	if not os.path.exists(test_save_dir):
		os.makedirs(test_save_dir)


	for cell in cells:
		run = runs[cell]
		model_path = os.path.join("data","exp_data","cla","6_cell_input_updated_100kb",\
				"gru",run)
		# test_data_dir = os.path.join("data","100kb_test_by_chr", cell)
		test_data_dir = os.path.join("data","100kb_cell_specific_regions",cell)
		cell_save_dir = os.path.join(test_save_dir,cell)

		if not os.path.exists(cell_save_dir):
			os.makedirs(cell_save_dir)




		save_file = os.path.join(cell_save_dir,"{}_test.csv".format(cell))
		test_loader = torch.load(os.path.join(test_data_dir,"{}_test.pth".format(cell)))
		pred, pred_binary = load_model_and_run(model_path,test_loader)

		with open(save_file,"w", newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(["cell","chrom","start","end","eigenvector","cla_label","mean_evec","region","model_pred","pred_binary"])

			for idx, batch in enumerate(test_loader):
				row = [cell,str(batch["chrom"])]
				row.append(int(batch["start_pos"]))
				row.append(int(batch["end_pos"]))
				row.append(float(batch["reg_labels"]))
				row.append(int(batch["cla_labels"]))
				row.append(float(batch["mean_evec"]))
				row.append(int(batch["region"]))
				row.append(str(pred[idx]))
				row.append(str(pred_binary[idx]))

				writer.writerow(row)


if __name__ == "__main__":
    main()


