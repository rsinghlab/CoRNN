import itertools
import os

cells = ["IMR90", "HMEC", "K562", "GM12878","HUVEC","NHEK"]
# cells = ["IMR90"]

parameters = {
	"layer": [1,2,3,4],
	"hidden" : [32,64,128],
}
param_list = parameters.keys()
param_vals = parameters.values()


for cell in cells:
	for param_selection in list(itertools.product(*param_vals)):
		params = {}
		
		command = 'python cnn/hm2ab.py --data_dir \
		"data/6_cell_input_updated/6_cell_input_updated_100kb/" --task "cla" \
		--model "gru" --split "5_cells" -Ts\
		 --epoch 10 --resolution "100kb" --cross_validation True --num_fold 5 \
		 --special_tag "100kb_GRU_update" --cell "{}"'.format(cell)

		for param_name, param_value in zip(param_list,param_selection):
			command = command +" --{} {}".format(param_name,param_value)

		os.system(command)

# cells = ["GM12878"]
# # cells = ["IMR90"]

# parameters = {
# 	"layer": [4],
# 	"hidden" : [32,64,128],
# }
# param_list = parameters.keys()
# param_vals = parameters.values()


# for cell in cells:
# 	for param_selection in list(itertools.product(*param_vals)):
# 		params = {}
		
# 		command = 'python cnn/hm2ab.py --data_dir \
# 		"data/6_cell_input_updated/6_cell_input_updated_100kb/" --task "cla" \
# 		--model "gru" --split "5_cells" -Ts\
# 		 --epoch 10 --resolution "100kb" --cross_validation True --num_fold 5 \
# 		 --special_tag "100kb_GRU_update" --cell "{}"'.format(cell)

# 		for param_name, param_value in zip(param_list,param_selection):
# 			command = command +" --{} {}".format(param_name,param_value)

# 		os.system(command)


# cells = ["NHEK", "HMEC", "K562", "HUVEC"]
# # cells = ["IMR90"]

# parameters = {
# 	"layer": [1,2,3,4],
# 	"hidden" : [32,64,128],
# }
# param_list = parameters.keys()
# param_vals = parameters.values()


# for cell in cells:
# 	for param_selection in list(itertools.product(*param_vals)):
# 		params = {}
		
# 		command = 'python cnn/hm2ab.py --data_dir \
# 		"data/6_cell_input_updated/6_cell_input_updated_100kb/" --task "cla" \
# 		--model "gru" --split "5_cells" -Ts\
# 		 --epoch 10 --resolution "100kb" --cross_validation True --num_fold 5 \
# 		 --special_tag "100kb_GRU_update" --cell "{}"'.format(cell)

# 		for param_name, param_value in zip(param_list,param_selection):
# 			command = command +" --{} {}".format(param_name,param_value)

# 		os.system(command)