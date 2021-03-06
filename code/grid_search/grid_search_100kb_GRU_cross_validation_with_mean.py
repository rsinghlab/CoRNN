import itertools
import os



cells = ["HUVEC", "HMEC","IMR90", "GM12878", "K562", "NHEK"]

parameters = {
	"layer": [1,2,3,4],
	"hidden" : [32,64,128],
}
param_list = parameters.keys()
param_vals = parameters.values()


for cell in cells:
	for param_selection in list(itertools.product(*param_vals)):
		params = {}
		
		command = 'python code/hm2ab.py --data_dir \
		"data/6_cell_input_updated/6_cell_input_updated_100kb/" --task "cla" \
		--model "gru" -Ts --epoch 10 --resolution "100kb" --cross_validation True --num_fold 5 \
		 --add_mean_evec True --cell "{}"'.format(cell)

		for param_name, param_value in zip(param_list,param_selection):
			command = command +" --{} {}".format(param_name,param_value)

		os.system(command)