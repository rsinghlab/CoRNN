
import matplotlib.pyplot as plt

import numpy as np

import datetime
import os



import matplotlib.pyplot as plt
from matplotlib import cm
# from colorspacious import cspace_converter
from collections import OrderedDict
from matplotlib import colors as mcolors

# cmaps = OrderedDict()
# colors = cmaps[]

def main():
	# parser = argparse.ArgumentParser()
	# parser.add_argument("-b", "--baseline", action="store_true",
 #                        help="add baseline")
	# parser.add_argument("-t", "--test", action="store_true",
 #                        help="add baseline")
	




	# # parser.add_argument('-m','--list', nargs='+', help='list of model type')
	# # parser.add_argument('-run','--list', nargs='+', help='None')
	# # parser.add_argument('-t','--list', nargs='+', help='list of model name')
	# args = parser.parse_args()

	# cells = ["IMR90","NHEK", "HMEC", "K562", "GM12878", "HUVEC"]
	# 100kb_models = {"100kb":"run_2021-03-24-09-09-38",\
	# 				 "50kb":"run_2021-04-18-18-33-08"
	# 				}


	now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	exp_dpath = os.path.join("data","exp_data","combined_curve","exp_{}".format(now))
	if not os.path.exists(exp_dpath):
		os.makedirs(exp_dpath)

	print(exp_dpath)


	# plotBarGraph_4(exp_dpath)
	# plotBarGraph_v2(exp_dpath)

	#plotBarGraph_three_bar(exp_dpath)
	# plotBarGraph_three_bar_noNHEK(exp_dpath)
	# plotBarGraph_four_bar_noNHEK_HUVEC(exp_dpath)
	# plotBarGraph_three_bar_noNHEK(exp_dpath)
	# cells = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	# # cells = ["GM12878"]
	# for cell in cells:
	# 	plotBarGraphFromData_100vs50(exp_dpath,cell)
	labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	data = {
		"100kb_5cells":[0.892,0.898,0.818,0.904,0.871,0.865], #0
		"100kb_mean":[0.877,0.881,0.792,0.946,0.939,0.946],
		"50kb_5cells":[0.898,0.889,0.803,0.911,0.886,0.880], #2
		"50kb_mean":[0.903,0.887,0.951,0.954,0.937,0.955],
		"50kb_gru":[0.924,0.904,0.838, 0.897, 0.880, 0.860],#4
		"50kb_bi_gru":[0.884,0.903,0.734,0.897,0.847,0.776],
		"50kb_single":[0.922,0.910,0.841,0.920,0.898,0.929],#6
		"100kb_single":[0.923,0.912,0.706,0.904,0.901,0.935],     
		"50kb_all":[0.914,0.942,0.832,0.903,0.918,0.817],#8
		"50kb_all_mean":[0.874,0.893,0.892,0.941,0.909,0.927],
		"100kb_all_mean":[0.872,0.893,0.850,0.938,0.905,0.916],#10
		"100kb_all":[0.925,0.940,0.697,0.916,0.926,0.775],
		"100kb_4cells":[0.911,0.936,0,0.929,0.912,0.771],
		"100kb_4cells_mean":[0.867,0.896,0,0.913,0.910,0.904],
		"100kb_3cells":[0.813,0.931,0,0.884,0.918,0],
		"100kb_3cells_mean":[0.834,0.889,0,0,903,0.906,0],
		"100kb_gru_add_mean_wrong":[0.891,0.947,0.850,0.959,0.948,0.936],
		"100kb_gru_add_mean":[0.901,0.938,0.857,0.957,0.953,0.916],
		"100kb_rf_600":[0.819,0.748,0.641,0.659,0.532,0.501],
		"100kb_rf_600_with_mean":[0.842,0.836,0.770,0.783,0.773,0.589],
		"100kb_rf_6":[0.812, 0.799,0.549, 0.684, 0.822, 0.508],
		"100kb_rf_6_with_mean":[0.837, 0.845, 0.784,0.878, 0.879, 0.776],
		"100kb_rf_12":[0.826, 0.785, 0.689, 0.723,0.790, 0.533],
		"100kb_rf_12_with_mean":[0.837, 0.852,0.788, 0.872,0.880,0.810],
		"100Kb_NHEK_GRU_region":[0.872,0.605,0.495,0.520,0.639,0.903],
		"100kb_NHEK_mean_region":[0.872,0.605,0.484,0.691,0.713,0.905],
		"100Kb_GM2878_GRU_region":[0.946,0.795,0.739,0.732,0.794,0.942],
		"100kb_GM2878_mean_region":[0.946,0.792,0.685,0.558,0.726,0.949],
		"100Kb_HMEC_GRU_region":[0.984,0.821,0.706,0.707,0.837,0.977],
		"100kb_HMEC_mean_region":[0.984,0.818,0.651,0.643,0.826,0.976],
		"100Kb_HUVEC_GRU_region":[0.957,0.760,0.614,0.588,0.737,0.976],
		"100kb_HUVEC_mean_region":[0.957,0.760,0.609,0.606,0.740,0.977],
		"100Kb_K562_GRU_region":[0.928,0.763,0.716,0.722,0.817,0.939],
		"100kb_K562_mean_region":[0.928,0.756,0.650,0.589,0.739,0.939],
		"100Kb_IMR90_GRU_region":[0.926,0.762,0.621,0.644,0.678,0.894],
		"100kb_IMR90_mean_region":[0.926,0.753,0.639,0.581,0.671,0.894],
		"100kb_linear_6":[0.859,0.934,0.786,0.828,0.875,0.900],
		"100kb_linear_7":[0.900,0.905,0.860,0.951,0.923,0.933],
		"100kb_logistic_6_default": [0.766,0.726,0.680,0.756,0.639,0.745],
		"100kb_logistic_7_default":[0.820,0.832,0.787,0.880,0.841,0.815],
		"100kb_logistic_12_default": [0.820,0.746,0.707,0.758,0.522,0.518],
		"100kb_logistic_13_default":[0.813,0.835,0.788,0.883,0.838,0.830],
		"100kb_logistic_12_gridsearch":[0.815,0.806,0.506,0.738,0.533,0.653],
		"100kb_logistic_13_gridsearch":[0.830,0.835,0.788,0.883,0.837,0.814],
		"GRU region": [0.9355, 0.751, 0.6485, 0.6522, 0.75, 0.9385],
		"Mean region": [0.9355, 0.747, 0.62, 0.611, 0.7358, 0.94],
		"CNN8_100kb": [0.892,0.898,0.818,0.904,0.871,0.865]
	}







	# labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	# data = {
	# 	"gru":[0.925,0.940,0.697,0.916,0.926,0.775],
	# 	"gru (add mean)":[0.891,0.947,0.850,0.959,0.948,0.936],
	# 	"mean evec":[0.872,0.893,0.850,0.938,0.905,0.916],
	# }
	# title = "100kb GRU vs GRU(add mean)"
	# plotBarGraph_three_bar_template(exp_dpath,labels,data,title)


	# labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	# # data = {
	# data = {
	# 	"100kb - gru - all cells":[0.901,0.938,0.857,0.957,0.953,0.916],
	# 	"100kb - mean - all cells":[0.872,0.893,0.850,0.938,0.905,0.916],
	# }
	# title = "100kb GRU - add mean evec"
	# plotBarGraph_two_bar_template(exp_dpath,labels,data,title)

	# labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	# # data = {
	# data = {
	# 	"gru (add mean)":[0.901,0.938,0.857,0.957,0.953,0.916],
	# 	"mean baseline":[0.872,0.893,0.850,0.938,0.905,0.916],
	# 	"RF 600":[0.819,0.748,0.641,0.659,0.532,0.501],
	# 	"RF 600 (add mean)":[0.842,0.836,0.770,0.783,0.773,0.589]
	# }
	# title = "100kb GRU vs Random Forest 600"
	# plotBarGraph_four_bar_template(exp_dpath,labels,data,title)

	# labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	# # data = {
	# data = {
	# 	"RF 600":[0.819,0.748,0.641,0.659,0.532,0.501],
	# 	"RF 600 (add mean)":[0.842,0.836,0.770,0.783,0.773,0.589],
	# 	"RF 6":[0.812, 0.799,0.549, 0.684, 0.822, 0.508],
	# 	"RF 6 (add mean)":[0.837, 0.845, 0.784,0.878, 0.879, 0.776]
	# }
	# title = "100kb Random Forest 600 vs 6 features"
	# colors = ["tab:brown","tab:pink","tab:cyan","tab:grey"]
	# plotBarGraph_four_bar_template(exp_dpath,labels,data,title, colors)


	# labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	# # data = {
	# data = {
	# 	"gru (add mean)":[0.901,0.938,0.857,0.957,0.953,0.916],
	# 	"mean baseline":[0.872,0.893,0.850,0.938,0.905,0.916],
	# 	"RF 6":[0.812, 0.799,0.549, 0.684, 0.822, 0.508],
	# 	"RF 6 + mean evec":[0.837, 0.845, 0.784,0.878, 0.879, 0.776]
	# }
	# title = "100kb GRU vs Random Forest 6 "
	# # colors = ["tab:blue","tab:orange","tab:cyan","tab:purple"]
	# plotBarGraph_four_bar_template(exp_dpath,labels,data,title)

	# labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	# # data = {
	# data = {
	# 	"RF 6":[0.812, 0.799,0.549, 0.684, 0.822, 0.508],
	# 	"RF 6 + mean evec":[0.837, 0.845, 0.784,0.878, 0.879, 0.776],
	# 	"RF 12":[0.826, 0.785, 0.689, 0.723,0.790, 0.533],
	# 	"RF 12 + mean evec":[0.837, 0.852,0.788, 0.872,0.880,0.810],
	# }
	# title = "100kb Random Forest 6 vs 12 features"
	# colors = ["tab:brown","tab:pink","tab:cyan","tab:grey"]
	# plotBarGraph_four_bar_template(exp_dpath,labels,data,title,colors)

	# labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	# # data = {
	# data = {
	# 	"gru (add mean)":[0.901,0.938,0.857,0.957,0.953,0.916],
	# 	"mean baseline":[0.872,0.893,0.850,0.938,0.905,0.916],
	# 	"RF 12":[0.826, 0.785, 0.689, 0.723,0.790, 0.533],
	# 	"RF 12 + mean evec":[0.837, 0.852,0.788, 0.872,0.880,0.810],
	# }
	# title = "100kb GRU vs Random Forest 12 "
	# # colors = ["tab:blue","tab:orange","tab:cyan","tab:purple"]
	# plotBarGraph_four_bar_template(exp_dpath,labels,data,title)

		
		

	# labels = ["0","1","2","3","4","5"]
	# # data = {
	# data = {
	# 	"GRU":[0.957,0.760,0.614,0.588,0.737,0.976],
	# 	"Mean baseline":[0.957,0.760,0.609,0.606,0.740,0.977],
	# }
	# title = "100kb HUVEC Test by Region (GRU vs Mean Baseline)"
	# # colors = ["tab:blue","tab:orange","tab:cyan","tab:purple"]
	# plotBarGraph_two_bar_template(exp_dpath,labels,data,title)

	# plot_12_bar()

	# labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]

	# data = {
	# 	"gru (add mean)":[0.901,0.938,0.857,0.957,0.953,0.916],
	# 	"mean baseline":[0.872,0.893,0.850,0.938,0.905,0.916],
	# 	"logistic Regression 12":[0.815,0.806,0.506,0.738,0.533,0.653],
	# 	"logistic Regression 12 + mean":[0.830,0.835,0.788,0.883,0.837,0.814]
	# }
	# title = "100kb GRU vs Logistic Regression "
	# # colors = ["tab:blue","tab:orange","tab:cyan","tab:purple"]
	# plotBarGraph_four_bar_template(exp_dpath,labels,data,title)

	# labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]

	# data = {
	# 	"gru (add mean)":[0.901,0.938,0.857,0.957,0.953,0.916],
	# 	"mean baseline":[0.872,0.893,0.850,0.938,0.905,0.916],
	# 	"RF 12":[0.826, 0.785, 0.689, 0.723,0.790, 0.533],
	# 	"RF 12 + mean":[0.837, 0.852,0.788, 0.872,0.880,0.810],
	# 	"logistic Regression 12":[0.815,0.806,0.506,0.738,0.533,0.653],
	# 	"logistic Regression 12 + mean":[0.830,0.835,0.788,0.883,0.837,0.814]
	# }
	# title = "100kb GRU vs baselines"
	# # colors = ["tab:blue","tab:orange","tab:cyan","tab:purple"]
	# plotBarGraph_six_bar_template(exp_dpath,labels,data,title)
	# labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	#
	# data = {
	# 	"GRU":[0.925,0.940,0.697,0.916,0.926,0.775],
	# 	"Mean Baseline":[0.872,0.893,0.850,0.938,0.905,0.916],#10
	# }
	# title = "100kb GRU vs Mean Eigenvectors"
	# # colors = ["tab:blue","tab:orange","tab:cyan","tab:purple"]
	# plotBarGraph_two_bar_template(exp_dpath,labels,data,title)



	# data = {
	# 	"Mean Baseline":[0.872,0.893,0.850,0.938,0.905,0.916],
	# 	"CoRNN (our model)":[0.901,0.938,0.857,0.957,0.953,0.916],
	# 	"GRU":[0.925,0.940,0.697,0.916,0.926,0.775],
	# 	"Random Forest":[0.826, 0.785, 0.689, 0.723,0.790, 0.533],
	# 	"Random Forest (add mean)":[0.837, 0.852,0.788, 0.872,0.880,0.810],
	# 	"Logistic Regression":[0.815,0.806,0.506,0.738,0.533,0.653],
	# 	"Logistic Regression (add mean)":[0.830,0.835,0.788,0.883,0.837,0.814]
	# }
	# labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	# title = "Testing Performance"
	# # colors = ["tab:blue","tab:orange","tab:cyan","tab:purple"]
	# plotBarGraph_seven_bar_final(exp_dpath,labels,data,title)

	# labels = ["0","1","2","3","4","5"]
	# # data = {
	# data = {
	# 	"GRU": [0.936, 0.751, 0.649, 0.652, 0.75, 0.939],
	# 	"Mean": [0.936, 0.747, 0.62, 0.611, 0.736, 0.94]
	# }
	# title = "100kb Test by Region (GRU vs Mean Baseline)"
	# # colors = ["tab:blue","tab:orange","tab:cyan","tab:purple"]
	# plotBarGraph_two_bar_template(exp_dpath,labels,data,title)


	# region_data = {
	# "NHEK_GRU":[0.872,0.605,0.495,0.520,0.639,0.903],
	# 	"NHEK_mean":[0.872,0.605,0.484,0.691,0.713,0.905],
	# 	"GM12878_GRU":[0.946,0.795,0.739,0.732,0.794,0.942],
	# 	"GM12878_mean":[0.946,0.792,0.685,0.558,0.726,0.949],
	# 	"HMEC_GRU":[0.984,0.821,0.706,0.707,0.837,0.977],
	# 	"HMEC_mean":[0.984,0.818,0.651,0.643,0.826,0.976],
	# 	"HUVEC_GRU":[0.957,0.760,0.614,0.588,0.737,0.976],
	# 	"HUVEC_mean":[0.957,0.760,0.609,0.606,0.740,0.977],
	# 	"K562_GRU":[0.928,0.763,0.716,0.722,0.817,0.939],
	# 	"K562_mean":[0.928,0.756,0.650,0.589,0.739,0.939],
	# 	"IMR90_GRU":[0.926,0.762,0.621,0.644,0.678,0.894],
	# 	"IMR90_mean":[0.926,0.753,0.639,0.581,0.671,0.894],
	# }

	# data = {
	# 	"Mean": [0.9342, 0.7487, 0.6208, 0.6099, 0.7373, 0.9387],
	# 	"GRU": [0.9342, 0.7532, 0.6121, 0.6860, 0.7643, 0.9387],
		
	# }

	# labels = ["0","1","2","3","4","5"]
	# title = "100kb Test by Region (GRU vs Mean Baseline)"
	# colors = ["tab:blue","tab:orange","tab:cyan","tab:purple"]
	# plotBarGraph_two_bar_error_bar_template(exp_dpath,labels,data,region_data, title)


	# data = {
	# 	"Mean": [0.936, 0.743, 0.616],
	# 	"GRU": [0.936, 0.759, 0.648],
	# }

	# labels = ["0:5","1:4","2:3"]
	# title = "100kb Test by Region (GRU vs Mean Baseline)"
	# colors = ["tab:blue","tab:orange","tab:cyan","tab:purple"]
	# plotBarGraph_two_bar_template(exp_dpath,labels,data, title)

	data = {
		"Mean": [0.936, 0.743, 0.616],
		"CoRNN": [0.936, 0.759, 0.648],
	}

	labels = ["0:5","1:4","2:3"]
	title = "Test Performance by Region"
	colors = ["tab:blue","tab:orange","tab:cyan","tab:purple"]
	plotBarGraph_two_bar_template(exp_dpath,labels,data, title)

	# data = {
	# 	"Mean Baseline":[0.872,0.893,0.850,0.938,0.905,0.916],
	# 	"CoRNN (our model)":[0.901,0.938,0.857,0.957,0.953,0.916],
	# 	"GRU":[0.925,0.940,0.697,0.916,0.926,0.775],
	# 	"Random Forest":[0.826, 0.785, 0.689, 0.723,0.790, 0.533],
	# 	"Random Forest (add mean)":[0.837, 0.852,0.788, 0.872,0.880,0.810],
	# 	"Logistic Regression":[0.815,0.806,0.506,0.738,0.533,0.653],
	# 	"Logistic Regression (add mean)":[0.830,0.835,0.788,0.883,0.837,0.814]
	# }
	# labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	# title = "Testing Performance"
	# # colors = ["tab:blue","tab:orange","tab:cyan","tab:purple"]
	# plotBarGraph_final_bar(exp_dpath,labels,data,title)
	
	
		

	
def plotBarGraph_two_bar_error_bar_template(exp_dpath, labels, data,region_data, title):
	img_path = os.path.join(exp_dpath,"100_50kb_cnn.png")

	mean_means = []
	mean_std = []
	gru_means = []
	gru_std = []

	cells = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]

	for region in range(6):
		mean_val = []
		gru_val = []
		for cell in cells:
			mean_val.append(region_data["{}_mean".format(cell)][region])
			gru_val.append(region_data["{}_GRU".format(cell)][region])
		mean_means.append(np.mean(mean_val))
		mean_std.append(np.std(mean_val))
		gru_means.append(np.mean(gru_val))
		gru_std.append(np.std(gru_val))







	
	x = np.arange(len(labels))  # the label locations
	width = 0.3  # the width of the bars

	print(len(list(data)))
		
	fig, ax = plt.subplots()
	bar_name = list(data.keys())
	bar_value = list(data.values())

	print(bar_name)
	print(bar_value)

	colors = ["tab:blue","tab:orange"]

	colors = [lighten_color("tab:blue",0.7),lighten_color("tab:orange",0.7)]

	rects1 = ax.bar(x - 0.5*width, bar_value[0],width,  \
		yerr = mean_std ,align='center', alpha=1, \
		ecolor='black',error_kw=dict(lw=1, capsize=4, capthick=1),\
		label=bar_name[0], color = colors[0])
	
	rects2 = ax.bar(x + 0.5*width, bar_value[1],width,  \
		yerr = gru_std,align='center', alpha=1, \
		ecolor='black', error_kw=dict(lw=1, capsize=4, capthick=1),\
		label=bar_name[1], color = colors[1])
	# rects4 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# rects1 = ax.bar(x - width, bar_value[2], width, label=bar_name[2])
	# rects2 = ax.bar(x, bar_value[4], width, label=bar_name[4])
	# rects3 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Testing Accuracy')
	ax.set_xlabel('Regions')
	ax.set_title(title)
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	plt.yticks(fontsize=10)
	ax.legend(loc='lower left', fontsize = 8)

	plt.ylim([0.5,1.0])

	# ax.bar_label(rects1, padding=-11, fontsize=8)
	# ax.bar_label(rects2, padding=-11, fontsize=8)
	# ax.bar_label(rects4, padding=3)

	fig.tight_layout()

	plt.show()
	plt.savefig(img_path,dpi=1000)
	plt.clf()



def plotBarGraph_two_bar_template(exp_dpath, labels, data, title):
	img_path = os.path.join(exp_dpath,"100_50kb_cnn.png")
	
	x = np.arange(len(labels))  # the label locations
	width = 0.3  # the width of the bars

	print(len(list(data)))
		
	fig, ax = plt.subplots()
	bar_name = list(data.keys())
	bar_value = list(data.values())

	print(bar_name)
	print(bar_value)

	colors = ["tab:blue","tab:orange"]

	colors = [lighten_color("tab:blue",0.7),lighten_color("tab:orange",0.7)]

	rects1 = ax.bar(x - 0.5*width, bar_value[0], width, label=bar_name[0], color = colors[0])
	rects2 = ax.bar(x + 0.5*width, bar_value[1], width, label=bar_name[1], color = colors[1])
	# rects4 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# rects1 = ax.bar(x - width, bar_value[2], width, label=bar_name[2])
	# rects2 = ax.bar(x, bar_value[4], width, label=bar_name[4])
	# rects3 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Testing Accuracy')
	ax.set_title(title)
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	plt.yticks(fontsize=10)
	ax.legend(loc='lower left', fontsize = 10)

	plt.ylim([0.5,1.0])

	ax.bar_label(rects1, padding=-11, fontsize=8)
	ax.bar_label(rects2, padding=-11, fontsize=8)
	# ax.bar_label(rects4, padding=3)

	fig.tight_layout()

	plt.show()
	plt.savefig(img_path,dpi=1000)
	plt.clf()


def plotBarGraph_three_bar_template(exp_dpath, labels, data, title):
	img_path = os.path.join(exp_dpath,"100_50kb_cnn.png")
	
	
	x = np.arange(len(labels))  # the label locations
	width = 0.25  # the width of the bars

	print(len(list(data)))
	
	
	fig, ax = plt.subplots(figsize=(10,5),dpi=150)
	bar_name = list(data.keys())
	bar_value = list(data.values())

	print(bar_name)
	print(bar_value)

 

	rects1 = ax.bar(x - width, bar_value[0], width, label=bar_name[0])
	rects2 = ax.bar(x, bar_value[1], width, label=bar_name[1])
	rects3 = ax.bar(x + width, bar_value[2], width, label=bar_name[2])
	# rects4 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# rects1 = ax.bar(x - width, bar_value[2], width, label=bar_name[2])
	# rects2 = ax.bar(x, bar_value[4], width, label=bar_name[4])
	# rects3 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Testing AUROC')
	ax.set_title(title)
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	
	plt.yticks(fontsize=10)
	ax.legend(loc='upper left', fontsize = 8)

	plt.ylim([0.65,1.0])

	ax.bar_label(rects1, padding=3, fontsize=8)
	ax.bar_label(rects2, padding=3, fontsize=8)
	ax.bar_label(rects3, padding=3, fontsize=8)
	# ax.bar_label(rects4, padding=3)

	fig.tight_layout()

	plt.show()
	plt.savefig(img_path)
	plt.clf()


def plotBarGraph_four_bar_template(exp_dpath, labels, data, title , colors = None):
	# 

	if colors == None:
		colors = ["tab:blue","tab:orange","tab:green","tab:red"]

	x = np.arange(len(labels))  # the label locations
	width = 0.2  # the width of the bars

	print(len(list(data)))
	
	
	fig, ax = plt.subplots(figsize=(10,5),dpi=120)
	bar_name = list(data.keys())
	bar_value = list(data.values())

	print(bar_name)
	print(bar_value)

 

	rects1 = ax.bar(x - 1.5*width, bar_value[0], width, label=bar_name[0],color = colors[0])
	rects2 = ax.bar(x - 0.5 * width, bar_value[1], width, label=bar_name[1],color = colors[1])
	rects3 = ax.bar(x + 0.5 * width, bar_value[2], width, label=bar_name[2],color = colors[2])
	rects4 = ax.bar(x + 1.5*width, bar_value[3], width, label=bar_name[3],color = colors[3])

	# rects1 = ax.bar(x - width, bar_value[2], width, label=bar_name[2])
	# rects2 = ax.bar(x, bar_value[4], width, label=bar_name[4])
	# rects3 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Testing AUROC')
	ax.set_title(title)
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	plt.yticks(fontsize=10)
	ax.legend(loc='lower left',fontsize = 10)

	plt.ylim([0.4,1.0])

	ax.bar_label(rects1, padding=3,fontsize=7)
	ax.bar_label(rects2, padding=3,fontsize=7)
	ax.bar_label(rects3, padding=3,fontsize=7)
	ax.bar_label(rects4, padding=3,fontsize=7)

	fig.tight_layout()

	plt.show()
	plt.savefig(img_path)
	plt.clf()


def plotBarGraph_six_bar_template(exp_dpath, labels, data, title , colors = None):
	# 
	if colors == None:
		colors = ["tab:blue","tab:orange","tab:green","tab:red",'tab:purple','tab:brown']

	x = np.arange(len(labels))  # the label locations
	width = 0.12  # the width of the bars

	print(len(list(data)))
	
	
	fig, ax = plt.subplots(figsize=(12,5),dpi=120)
	bar_name = list(data.keys())
	bar_value = list(data.values())

	print(bar_name)
	print(bar_value)

 

	rects1 = ax.bar(x - 2.5*width, bar_value[0], width, label=bar_name[0],color = colors[0])
	rects2 = ax.bar(x - 1.5 * width, bar_value[1], width, label=bar_name[1],color = colors[1])
	rects3 = ax.bar(x - 0.5 * width, bar_value[2], width, label=bar_name[2],color = colors[2])
	rects4 = ax.bar(x + 0.5*width, bar_value[3], width, label=bar_name[3],color = colors[3])
	rects5 = ax.bar(x + 1.5*width, bar_value[4], width, label=bar_name[4],color = colors[4])
	rects6 = ax.bar(x + 2.5*width, bar_value[5], width, label=bar_name[5],color = colors[5])

	# rects1 = ax.bar(x - width, bar_value[2], width, label=bar_name[2])
	# rects2 = ax.bar(x, bar_value[4], width, label=bar_name[4])
	# rects3 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Testing AUROC')
	ax.set_title(title)
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	plt.yticks(fontsize=10)
	ax.legend(loc='lower left',fontsize = 8)

	plt.ylim([0.4,1.0])

	ax.bar_label(rects1, padding=3,fontsize=6)
	ax.bar_label(rects2, padding=-3,fontsize=6)
	ax.bar_label(rects3, padding=3,fontsize=6)
	ax.bar_label(rects4, padding=3,fontsize=6)
	ax.bar_label(rects5, padding=3,fontsize=6)
	ax.bar_label(rects6, padding=3,fontsize=6)

	fig.tight_layout()

	plt.show()
	plt.savefig(img_path)
	plt.clf()

def plotBarGraph_final_bar(exp_dpath, labels, data, title):
	labels_order = ["GM12878", "K562", "IMR90","HMEC", "NHEK", "HUVEC"]
	img_path = os.path.join(exp_dpath,"100_50kb_cnn.png")
	
	x = np.arange(len(labels))  # the label locations
	width = 0.1 # the width of the bars

	print(len(list(data)))
		
	fig, ax = plt.subplots(figsize=(12,5))
	bar_name = list(data.keys())
	bar_value = list(data.values())
	bar_value = np.array(bar_value)
	val_by_cell = {}


	for i, label in enumerate(labels):
		val_by_cell[label] = bar_value[:,i]

	print(val_by_cell)
	bar_val_updated = np.zeros((7,6))
	
	for i, label in enumerate(labels_order):
		bar_val_updated[:,i] = val_by_cell[label]
		
	print(bar_val_updated)
	bar_value = bar_val_updated


	colors = ['#88BEE7','#F0B67F','#8CC084','#9180C6','#D47395','#87d4d4','#F9D88A',]

	rects1 = ax.bar(x - 3*width, bar_value[0], width, label=bar_name[0],color = colors[0], edgecolor='white')
	rects2 = ax.bar(x - 2*width, bar_value[1], width, label=bar_name[1],color = colors[1], edgecolor='white')
	rects3 = ax.bar(x - width, bar_value[2], width, label=bar_name[2],color = colors[2], edgecolor='white')
	rects4 = ax.bar(x, bar_value[3], width, label=bar_name[3],color = colors[3], edgecolor='white')
	rects5 = ax.bar(x + width, bar_value[4], width, label=bar_name[4],color = colors[4], edgecolor='white')
	rects6 = ax.bar(x + 2*width, bar_value[5], width, label=bar_name[5],color = colors[5], edgecolor='white')
	rects7 = ax.bar(x + 3*width, bar_value[6], width, label=bar_name[6],color = colors[6], edgecolor='white')
	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Testing AUROC')
	ax.set_title(title)
	ax.set_xticks(x)
	ax.set_xticklabels(labels_order)
	plt.yticks(fontsize=10)
	ax.legend(loc='lower left',fontsize = 6)

	plt.ylim([0.3,1.0])

	bar_label_font_size = 4

	ax.bar_label(rects1, padding=3,fontsize=bar_label_font_size)
	ax.bar_label(rects2, padding=3,fontsize=bar_label_font_size)
	ax.bar_label(rects3, padding=3,fontsize=bar_label_font_size)
	ax.bar_label(rects4, padding=3,fontsize=bar_label_font_size)
	ax.bar_label(rects5, padding=3,fontsize=bar_label_font_size)
	ax.bar_label(rects6, padding=3,fontsize=bar_label_font_size)
	ax.bar_label(rects7, padding=3,fontsize=bar_label_font_size)

	# ax.bar_label(rects4, padding=3)

	fig.tight_layout()

	plt.show()
	img_path = os.path.join(".","data","CoRNN-vs-baselines.eps")
	plt.savefig(img_path)
	plt.clf()

def plotBarGraph_seven_bar_final(exp_dpath, labels, data, title , colors = None):

	labels_order = ["GM12878", "K562", "IMR90","HMEC", "NHEK", "HUVEC"]
	# 
	if colors == None:
		colors = ["tab:blue","tab:orange","tab:green","tab:red",'tab:purple','tab:brown']

	x = np.arange(len(labels))  # the label locations
	width = 0.1  # the width of the bars

	print(len(list(data)))
	
	
	fig, ax = plt.subplots(figsize=(12,5))
	bar_name = list(data.keys())
	bar_value = list(data.values())

	print(bar_name)
	print(bar_value)

	colors = ['#88BEE7','#F0B67F','#8CC084',\
	'#9180C6','#D47395','#87d4d4','#F9D88A',]

	bar_value = np.array(bar_value)
	val_by_cell = {}


	for i, label in enumerate(labels):
		val_by_cell[label] = bar_value[:,i]

	print(val_by_cell)

	bar_val_updated = np.zeros((7,6))
	
	for i, label in enumerate(labels_order):
		bar_val_updated[:,i] = val_by_cell[label]
		
	print(bar_val_updated)
	bar_value = bar_val_updated
	# bar_name = labels_order
	print("updated")
	print(bar_name)
	print(bar_value)

	rects1 = ax.bar(x - 3*width, bar_value[0], width, label=bar_name[0],color = colors[0], edgecolor='white')
	rects2 = ax.bar(x - 2*width, bar_value[1], width, label=bar_name[1],color = colors[1], edgecolor='white')
	rects3 = ax.bar(x - width, bar_value[2], width, label=bar_name[2],color = colors[2], edgecolor='white')
	rects4 = ax.bar(x, bar_value[3], width, label=bar_name[3],color = colors[3], edgecolor='white')
	rects5 = ax.bar(x + width, bar_value[4], width, label=bar_name[4],color = colors[4], edgecolor='white')
	rects6 = ax.bar(x + 2*width, bar_value[5], width, label=bar_name[5],color = colors[5], edgecolor='white')
	rects7 = ax.bar(x + 3*width, bar_value[6], width, label=bar_name[6],color = colors[6], edgecolor='white')


	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Testing AUROC')
	ax.set_title(title)
	ax.set_xticks(x)
	ax.set_xticklabels(labels_order)
	plt.yticks(fontsize=10)
	ax.legend(loc='lower left',fontsize = 6)

	plt.ylim([0.4,1.0])

	ax.bar_label(rects1, padding=3,fontsize=6)
	ax.bar_label(rects2, padding=1,fontsize=6)
	ax.bar_label(rects3, padding=7,fontsize=6)
	ax.bar_label(rects4, padding=3,fontsize=6)
	ax.bar_label(rects5, padding=3,fontsize=6)
	ax.bar_label(rects6, padding=3,fontsize=6)
	ax.bar_label(rects7, padding=3,fontsize=6)

	fig.tight_layout()

	plt.show()
	img_path = os.path.join(".","data","CoRNN-vs-baselines.eps")
	plt.savefig(img_path,dpi=1000)
	plt.clf()

def plot_12_bar():

	
	data = {
	   "gru":[0.926, 0.928, 0.872, 0.984, 0.946, 0.957],
	   "mean baseline":[0.926, 0.928, 0.872, 0.984, 0.946, 0.957],
       "gru_1":[0.762, 0.763, 0.605, 0.821, 0.795, 0.76 ],
       "mean_1":[0.753, 0.756, 0.605, 0.818, 0.792, 0.76 ],
       "gru_2":[0.621, 0.716, 0.495, 0.706, 0.739, 0.614],
       "mean_2":[0.639, 0.65 , 0.484, 0.651, 0.685, 0.609],
       "gru_3":[0.644, 0.722, 0.52 , 0.707, 0.732, 0.588],
       "mean_3":[0.581, 0.589, 0.691, 0.643, 0.558, 0.606],
       "gru_4":[0.678, 0.817, 0.639, 0.837, 0.794, 0.737],
       "mean_4":[0.671, 0.739, 0.713, 0.826, 0.726, 0.74 ],
       "gru_5":[0.894, 0.939, 0.903, 0.977, 0.942, 0.976],  
       "mean_5":[0.894, 0.939, 0.905, 0.976, 0.949, 0.977]
	}
	labels = ["\nIMR90","\nK562","\nNHEK","\nHMEC","\nGM12878","\nHUVEC"]
	x = np.arange(len(labels))  # the label locations
	width = 0.07  # the width of the bars

	fig, ax = plt.subplots(figsize=(10,5),dpi=120)
	bar_name = list(data.keys())
	bar_value = list(data.values())

	colors = [lighten_color("tab:blue",0.7),lighten_color("tab:orange",0.7)]



	rects1 = ax.bar(x - 5.8*width, bar_value[0], width, label=bar_name[0],color = colors[0])
	rects2 = ax.bar(x - 4.8*width, bar_value[1], width, label=bar_name[1],color = colors[1])
	rects3 = ax.bar(x - 3.7*width, bar_value[2], width, color = colors[0])
	rects4 = ax.bar(x - 2.7*width, bar_value[3], width, color = colors[1])
	rects5 = ax.bar(x - 1.6*width, bar_value[4], width, color = colors[0])
	rects6 = ax.bar(x - 0.6 * width, bar_value[5], width, color = colors[1])
	rects7 = ax.bar(x + 0.5 * width, bar_value[6], width, color = colors[0])
	rects8 = ax.bar(x + 1.5*width, bar_value[7], width, color = colors[1])
	rects9 = ax.bar(x + 2.6*width, bar_value[8], width, color = colors[0])
	rects10 = ax.bar(x + 3.6*width, bar_value[9], width, color = colors[1])
	rects11 = ax.bar(x + 4.7*width, bar_value[10], width, color = colors[0])
	rects12 = ax.bar(x + 5.7*width, bar_value[11], width, color = colors[1])

	ax.set_ylabel('Testing Accuracy')
	ax.set_title("100kb Test by Region (GRU vs Mean Baseline)")

	labels = ["0","1","2","\nIMR90","3","4","5",\
		"0","1","2","\nK562","3","4","5",\
		"0","1","2","\nNHEK","3","4","5",\
		"0","1","2","\nHMEC","3","4","5",\
		"0","1","2","\nGM12878","3","4","5",\
		"0","1","2","\nHUVEC","3","4","5",]
	ticks_pos = []
	for i in x:
		ticks_pos.append(i-5.3*width)
		ticks_pos.append(i-3.2*width)
		ticks_pos.append(i-1.1*width)
		ticks_pos.append(i)
		ticks_pos.append(i+1.0*width)
		ticks_pos.append(i+3.1*width)
		ticks_pos.append(i+5.2*width)
		

	ax.set_xticks(ticks_pos)
	ax.set_xticklabels(labels)

	plt.yticks(fontsize=10)
	ax.legend(loc='lower left',fontsize = 10)

	plt.ylim([0.45,1.0])

	ax.bar_label(rects1, padding=3,fontsize=3)
	ax.bar_label(rects2, padding=0,fontsize=3)
	ax.bar_label(rects3, padding=3,fontsize=3)
	ax.bar_label(rects4, padding=0,fontsize=3)
	ax.bar_label(rects5, padding=3,fontsize=3)
	ax.bar_label(rects6, padding=3,fontsize=3)
	ax.bar_label(rects7, padding=3,fontsize=3)
	ax.bar_label(rects8, padding=3,fontsize=3)
	ax.bar_label(rects9, padding=3,fontsize=3)
	ax.bar_label(rects10, padding=3,fontsize=3)
	ax.bar_label(rects11, padding=3,fontsize=3)
	ax.bar_label(rects12, padding=0,fontsize=3)

	fig.tight_layout()

	plt.show()
	plt.savefig(img_path)
	plt.clf()

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def plotBarGraph_v2(exp_dpath):
	x = ["cnn(19-22)","gru(19-22)","mean(19-22)","gru(all)","mean(all)"]
	y = [0.866,0.880,0.937,0.918,0.909]

	fig, ax = plt.subplots()    
	width = 0.75 # the width of the bars 
	ind = np.arange(len(y))  # the x locations for the groups
	plt.ylim([0.70,1.0])
	ax.bar(ind, y, width, color="green")
	ax.set_xticks(ind)
	ax.set_xticklabels(x, minor=False)
	for i, v in enumerate(y):
		ax.text(i-0.2, v + .01, str(v), color='black')
	plt.xlabel("Experiments")
	plt.ylabel("Testing AUROC")
	plt.title("GM12878")
  
	plt.show()
	plt.savefig(os.path.join('test.png'), dpi=300, format='png', bbox_inches='tight')

def plotBarGraph(exp_dpath):
	x = ["cnn(19-22)","gru(19-22)","mean(19-22)","gru(all)","mean(all)"]
	data = [0.866,0.880,0.937,0.918,0.909]

	x_pos = [i for i, _ in enumerate(x)]

	plt.figure(figsize=(6,5))
	plt.ylim([0.70,1.0])

	plt.bar(x_pos, data, color='green')
	for i, v in enumerate(data):
		plt.text(v, i + .1, str(v), color = 'black', fontweight = 'bold')

	plt.xlabel("Experiments")
	plt.ylabel("Testing AUROC")
	plt.title("50kb GM12878")

	plt.xticks(x_pos, x)

	plt.show()

def plotBarGraphFromData_100kb(exp_dpath,cell):
	cells = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	all_data = {
		"100kb_5cells":[0.892,0.898,0.818,0.904,0.871,0.865],
		"100kb_single":[0.923,0.912,0.706,0.904,0.901,0.935], 
		"100kb_all_mean":[0.872,0.893,0.850,0.938,0.905,0.916],
		"100kb_all":[0,0,0,0,0.911,0],
	}

	x = ["cnn(19-22)","mean(19-22)","gru(all)","mean(all)"]

	index = cells.index(cell)

	y = [value_list[index] for value_list in list(all_data.values())]

	fig, ax = plt.subplots(figsize=(8,5))    
	width = 0.75 # the width of the bars 
	ind = np.arange(len(y))  # the x locations for the groups
	plt.ylim([0.7,1.0])
	ax.bar(ind, y, width, color="green")
	ax.set_xticks(ind)
	ax.set_xticklabels(x, minor=False)
	for i, v in enumerate(y):
		ax.text(i-0.2, v + .01, str(v), color='black')
	plt.xlabel("Experiments")
	plt.ylabel("Testing AUROC")
	plt.title("100kb {}".format(cell))
  
	plt.show()
	# plt.savefig(os.path.join('{}.png'.format(cell)), dpi=300, format='png', bbox_inches='tight')
	plt.savefig(os.path.join('{}.png'.format(cell)))


def plotBarGraphFromData_100vs50(exp_dpath,cell):

	cells = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	all_data = {
		"50kb_all":[0.914,0.942,0.832,0.903,0.918,0.817],
		"50kb_all_mean":[0.874,0.893,0.892,0.941,0.909,0.927],
		"100kb_all":[0.908,0.939,0.823,0.919,0.911,0.883],
		"100kb_all_mean":[0.872,0.893,0.850,0.938,0.905,0.916],
	}

	x = ["50kb-gru(all)","50kb-mean(all)","100kb-gru(all)","100kb-mean(all)"]

	index = cells.index(cell)

	y = [value_list[index] for value_list in list(all_data.values())]

	fig, ax = plt.subplots(figsize=(8,5))    
	width = 0.75 # the width of the bars 
	ind = np.arange(len(y))  # the x locations for the groups
	plt.ylim([0.8,0.95])
	ax.bar(ind, y, width, color="green")
	ax.set_xticks(ind)
	ax.set_xticklabels(x, minor=False)
	for i, v in enumerate(y):
		ax.text(i-0.2, v + .01, str(v), color='black')
	plt.xlabel("Experiments")
	plt.ylabel("Testing AUROC")
	plt.title("100kb vs 50kb: {}".format(cell))
  
	plt.show()
	# plt.savefig(os.path.join('{}.png'.format(cell)), dpi=300, format='png', bbox_inches='tight')
	plt.savefig(os.path.join('{}.png'.format(cell)))


def plotBarGraphFromData_50kb(exp_dpath	,cell):

	cells = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	all_data = {
		"50kb_5cells":[0.898,0.889,0.803,0.911,0.886,0.880],
		"50kb_gru":[0.924,0.904,0.838, 0.897, 0.880, 0.860],
		"50kb_mean":[0.903,0.887,0.951,0.954,0.937,0.955],
		"50kb_all":[0.914,0.942,0.832,0.903,0.918,0.817],
		"50kb_all_mean":[0.874,0.893,0.892,0.941,0.909,0.927]
	}

	x = ["cnn(19-22)","gru(19-22)","mean(19-22)","gru(all)","mean(all)"]

	index = cells.index(cell)

	y = [value_list[index] for value_list in list(all_data.values())]

	fig, ax = plt.subplots(figsize=(8,5))    
	width = 0.75 # the width of the bars 
	ind = np.arange(len(y))  # the x locations for the groups
	plt.ylim([0.60,1.0])
	ax.bar(ind, y, width, color="green")
	ax.set_xticks(ind)
	ax.set_xticklabels(x, minor=False)
	for i, v in enumerate(y):
		ax.text(i-0.2, v + .01, str(v), color='black')
	plt.xlabel("Experiments")
	plt.ylabel("Testing AUROC")
	plt.title("50kb {}".format(cell))
  
	plt.show()
	# plt.savefig(os.path.join('{}.png'.format(cell)), dpi=300, format='png', bbox_inches='tight')
	plt.savefig(os.path.join('{}.png'.format(cell)))


def plotBarGraph_1(exp_dpath):
	
	labels = ["cnn(19-22)","gru(19-22)","mean(19-22)","gru(all)","mean(all)"]
	data = [0.866,0.880,0.937,0.918,0.909]
	fig = plt.figure()
	ax = fig.add_axes([0,0,1,1])
	# langs = ['C', 'C++', 'Java', 'Python', 'PHP']
	# students = [23,17,35,29,12]
	x = np.arange(len(labels))
	ax.bar(labels,data)
	ax.set_ylabel('Testing AUROC')
	ax.set_title("50kb GM12878")
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	plt.yticks(fontsize=10)
	ax.legend(loc='upper left')
	plt.show()

	img_path = os.path.join(exp_dpath,"GM12878.png")
	plt.savefig(img_path)
	plt.clf()

def plotBarGraph_two_bar(exp_dpath, labels = None, data = None):
	img_path = os.path.join(exp_dpath,"100_50kb_cnn.png")
	
	labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	data = {
		"100kb_5cells":[0.892,0.898,0.818,0.904,0.871,0.865], #0
		"100kb_mean":[0.877,0.881,0.792,0.946,0.939,0.946],
		"50kb_5cells":[0.898,0.889,0.803,0.911,0.886,0.880], #2
		"50kb_mean":[0.903,0.887,0.951,0.954,0.937,0.955],
		"50kb_gru":[0.924,0.904,0.838, 0.897, 0.880, 0.860],#4
		"50kb_bi_gru":[0.884,0.903,0.734,0.897,0.847,0.776],
		"50kb_single":[0.922,0.910,0.841,0.920,0.898,0.929],#6
		"100kb_single":[0.923,0.912,0.706,0.904,0.901,0.935], 
		"50kb_all":[0.914,0.942,0.832,0.903,0.918,0.817],#8
		"50kb_all_mean":[0.874,0.893,0.892,0.941,0.909,0.927],
		"100kb_all_mean":[0.872,0.893,0.850,0.938,0.905,0.916],#10
		"50kb_all":[0.914,0.942,0.832,0.903,0.918,0.817],
		"100kb_all":[0.908,0.939,0.823,0.919,0.911,0.883],
		"100kb_all_noNHEK":[0.911,0.936,0,0.929,0.912,0.847],
		"50kb_all_noNHEK":[0.911,0.866,0,0.871,0.90,0.800],
		"50kb_all_noNHEK_mean":[0.865,0.895,0,0.912,0.911,0.925]
	}

	x = np.arange(len(labels))  # the label locations
	width = 0.4  # the width of the bars

	print(len(list(data)))
	
	
	fig, ax = plt.subplots(figsize=(8,5))
	bar_name = list(data.keys())
	bar_value = list(data.values())

	print(bar_name)
	print(bar_value)

 

	rects1 = ax.bar(x - width/2, bar_value[11], width, label=bar_name[11])
	# rects2 = ax.bar(x - width/2, bar_value[1], width, label=bar_name[1])
	rects3 = ax.bar(x + width/2, bar_value[10], width, label=bar_name[10])
	# rects4 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# rects1 = ax.bar(x - width, bar_value[2], width, label=bar_name[2])
	# rects2 = ax.bar(x, bar_value[4], width, label=bar_name[4])
	# rects3 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Testing AUROC')
	ax.set_title('100kb all chromosome, GRU vs Mean Eigenvectors')
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	plt.yticks(fontsize=10)
	ax.legend(loc='upper left')

	plt.ylim([0.60,1.0])

	ax.bar_label(rects1, padding=3)
	# ax.bar_label(rects2, padding=3)
	ax.bar_label(rects3, padding=3)
	# ax.bar_label(rects4, padding=3)

	fig.tight_layout()

	plt.show()


def plotBarGraph_three_bar(exp_dpath, labels = None, data = None):
	img_path = os.path.join(exp_dpath,"100_50kb_cnn.png")
	
	labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	data = {
		"100kb_5cells":[0.892,0.898,0.818,0.904,0.871,0.865], #0
		"100kb_mean":[0.877,0.881,0.792,0.946,0.939,0.946],
		"50kb_5cells":[0.898,0.889,0.803,0.911,0.886,0.880], #2
		"50kb_mean":[0.903,0.887,0.951,0.954,0.937,0.955],
		"50kb_gru":[0.924,0.904,0.838, 0.897, 0.880, 0.860],#4
		"50kb_bi_gru":[0.884,0.903,0.734,0.897,0.847,0.776],
		"50kb_single":[0.922,0.910,0.841,0.920,0.898,0.929],#6
		"100kb_single":[0.923,0.912,0.706,0.904,0.901,0.935], 
		"50kb_all":[0.914,0.942,0.832,0.903,0.918,0.817],#8
		"50kb_all_mean":[0.874,0.893,0.892,0.941,0.909,0.927],
		"100kb_all_mean":[0.872,0.893,0.850,0.938,0.905,0.916],#10
		"100kb_all_wrong":[0.908,0.939,0.823,0.919,0.911,0.883],
		"100kb_all":[0.760,0.940,0.697,0.916,0.926,0.775],
		"100kb_4cells":[0.911,0.936,0,0.929,0.912,0.771],
		"100kb_4cells_mean":[0.867,0.896,0,0.913,0.910,0.904],
		"100kb_3cells":[0.813,0.931,0,0.884,0.918,0],
		"100kb_3cells_mean":[0.834,0.889,0,0,903,0.906,0]
	}

	x = np.arange(len(labels))  # the label locations
	width = 0.25  # the width of the bars

	print(len(list(data)))
	
	
	fig, ax = plt.subplots(figsize=(8,5))
	bar_name = list(data.keys())
	bar_value = list(data.values())

	print(bar_name)
	print(bar_value)

 

	rects1 = ax.bar(x - width, bar_value[11], width, label=bar_name[11])
	rects2 = ax.bar(x, bar_value[12], width, label=bar_name[12])
	rects3 = ax.bar(x + width, bar_value[10], width, label=bar_name[10])
	# rects4 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# rects1 = ax.bar(x - width, bar_value[2], width, label=bar_name[2])
	# rects2 = ax.bar(x, bar_value[4], width, label=bar_name[4])
	# rects3 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Testing AUROC')
	ax.set_title('100kb all chromosome, GRU vs Mean Eigenvectors')
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	plt.yticks(fontsize=10)
	ax.legend(loc='upper left')

	plt.ylim([0.75,1.0])

	ax.bar_label(rects1, padding=-15)
	ax.bar_label(rects2, padding=3)
	ax.bar_label(rects3, padding=3)
	# ax.bar_label(rects4, padding=3)

	fig.tight_layout()

	plt.show()
	plt.savefig(img_path)
	plt.clf()

def plotBarGraph_four_bar(exp_dpath, labels = None, data = None):
	img_path = os.path.join(exp_dpath,"100_50kb_cnn.png")
	
	labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	data = {
		"50kb_all":[0.914,0.942,0.832,0.903,0.918,0.817],#0
		"50kb_all_mean":[0.874,0.893,0.892,0.941,0.909,0.927],
		"100kb_5cells":[0.892,0.898,0.818,0.904,0.871,0.865], #2
		"100kb_mean":[0.877,0.881,0.792,0.946,0.939,0.946], #3
		"100kb_single":[0.923,0.912,0.706,0.904,0.901,0.935],  #4
		"100kb_all_mean":[0.872,0.893,0.850,0.938,0.905,0.916],#5
		"100kb_all":[0.760,0.940,0.697,0.916,0.926,0.775],#6
		"100kb_4cells":[0.911,0.936,0,0.929,0.912,0.771],#7
		"100kb_4cells_mean":[0.867,0.896,0,0.913,0.910,0.904],
		"100kb_3cells":[0.813,0.931,0,0.884,0.918,0],
		"100kb_3cells_mean":[0.834,0.889,0,0.903,0.906,0]
	}
	labels = ["IMR90","K562", "HMEC", "GM12878"]
	data = {
		"100kb_4cells":[0.911,0.936,0.929,0.912],
		"100kb_4cells_mean":[0.867,0.896,0.913,0.910],
		"100kb_3celss":[0.813,0.931,0.884,0.918],
		"100kb_3cells_mean":[0.834,0.889,0.903,0.906]
	}

	labels = ["IMR90","K562", "HMEC", "GM12878", "HUVEC"]

	data = {
		"50k-GRU":[0.914,0.942,0.903,0.918,0.817],#8
		"50kb_mean":[0.874,0.893,0.941,0.909,0.927],
		"50kb-GRU-noNHEK":[0.911,0.866,0.871,0.90,0.800],
		"50kb_noNHEK_mean":[0.865,0.895,0.912,0.911,0.925]
	}

	

	x = np.arange(len(labels))  # the label locations
	width = 0.2  # the width of the bars

	print(len(list(data)))
	
	
	fig, ax = plt.subplots(figsize=(8,5))
	bar_name = list(data.keys())
	bar_value = list(data.values())

	print(bar_name)
	print(bar_value)

 

	rects1 = ax.bar(x - 1.5*width, bar_value[0], width, label=bar_name[0])
	rects2 = ax.bar(x - 0.5 * width, bar_value[1], width, label=bar_name[1])
	rects3 = ax.bar(x + 0.5 * width, bar_value[2], width, label=bar_name[2])
	rects4 = ax.bar(x + 1.5*width, bar_value[3], width, label=bar_name[3])

	# rects1 = ax.bar(x - width, bar_value[2], width, label=bar_name[2])
	# rects2 = ax.bar(x, bar_value[4], width, label=bar_name[4])
	# rects3 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Testing AUROC')
	ax.set_title('50kb GRU - remove NHEK')
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	plt.yticks(fontsize=10)
	ax.legend(loc='upper left',fontsize=8)

	plt.ylim([0.65,1.0])

	ax.bar_label(rects1, padding=3,fontsize=8)
	ax.bar_label(rects2, padding=3,fontsize=8)
	ax.bar_label(rects3, padding=3,fontsize=8)
	ax.bar_label(rects4, padding=3,fontsize=8)

	fig.tight_layout()

	plt.show()
	plt.savefig(img_path)
	plt.clf()

def plotBarGraph_five_bar(exp_dpath, labels = None, data = None):
	img_path = os.path.join(exp_dpath,"100_50kb_cnn.png")
	
	labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	data = {
		"50kb_all":[0.914,0.942,0.832,0.903,0.918,0.817],#0
		"50kb_all_mean":[0.874,0.893,0.892,0.941,0.909,0.927],
		"100kb_5cells":[0.892,0.898,0.818,0.904,0.871,0.865], #2
		"100kb_mean":[0.877,0.881,0.792,0.946,0.939,0.946], #3
		"100kb_single":[0.923,0.912,0.706,0.904,0.901,0.935],  #4
		"100kb_all_mean":[0.872,0.893,0.850,0.938,0.905,0.916],#5
		"100kb_all":[0.760,0.940,0.697,0.916,0.926,0.775],#6
		"100kb_4cells":[0.911,0.936,0,0.929,0.912,0.771],#7
		"100kb_4cells_mean":[0.867,0.896,0,0.913,0.910,0.904],
		"100kb_3cells":[0.813,0.931,0,0.884,0.918,0],
		"100kb_3cells_mean":[0.834,0.889,0,0.903,0.906,0]
	}

	labels = ["IMR90","K562", "HMEC", "GM12878", "HUVEC"]

	data = {
		"50kb-GRU-best":[0.914,0.942,0.903,0.918,0.817],#8
		"50kb-GRU-worst":[0.92,0.93,0.90,0.84,0.81],
		"50kb_mean":[0.874,0.893,0.941,0.909,0.927],
		"50kb-GRU-noNHEK":[0.911,0.866,0.871,0.90,0.800],
		"50kb_noNHEK_mean":[0.865,0.895,0.912,0.911,0.925],
	}

	x = np.arange(len(labels))  # the label locations
	width = 0.2  # the width of the bars

	print(len(list(data)))
	
	
	fig, ax = plt.subplots(figsize=(8,5))
	bar_name = list(data.keys())
	bar_value = list(data.values())

	print(bar_name)
	print(bar_value)

 

	rects1 = ax.bar(x - 1.5*width, bar_value[0], width, label=bar_name[0])
	rects2 = ax.bar(x - 0.5 * width, bar_value[1], width, label=bar_name[1])
	rects3 = ax.bar(x + 0.5 * width, bar_value[2], width, label=bar_name[2])
	rects4 = ax.bar(x + 1.5*width, bar_value[3], width, label=bar_name[3])

	# rects1 = ax.bar(x - width, bar_value[2], width, label=bar_name[2])
	# rects2 = ax.bar(x, bar_value[4], width, label=bar_name[4])
	# rects3 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Testing AUROC')
	ax.set_title('50kb GRU - remove NHEK')
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	plt.yticks(fontsize=10)
	ax.legend(loc='upper left',fontsize=8)

	plt.ylim([0.65,1.0])

	ax.bar_label(rects1, padding=3,fontsize=8)
	ax.bar_label(rects2, padding=3,fontsize=8)
	ax.bar_label(rects3, padding=3,fontsize=8)
	ax.bar_label(rects4, padding=3,fontsize=8)

	fig.tight_layout()

	plt.show()
	plt.savefig(img_path)
	plt.clf()


def plotBarGraph_three_bar_noNHEK(exp_dpath, labels = None, data = None):
	img_path = os.path.join(exp_dpath,"100_50kb_cnn.png")
	
	labels = ["IMR90","K562", "HMEC", "GM12878", "HUVEC"]
	data = {
		"100kb_all_mean":[0.872,0.893,0.938,0.905,0.916],#10
		"100kb_all":[0.908,0.939,0.919,0.911,0.883],
		"100kb_all_noNHEK":[0.911,0.936,0.929,0.912,0.847],
	}


	data = {
		"50kb-GRU-best":[0.914,0.942,0.903,0.918,0.817],#8
		"50kb-GRU-worst":[0.92,0.93,0.90,0.84,0.81],
		"50kb_mean":[0.874,0.893,0.941,0.909,0.927]
	}

	labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	data = {
		"gru-all":[0.760,0.940,0.697,0.916,0.926,0.775],#6
		"gru-add mean":[0.904,0.954,0.963,0.954,0.956,0.912],
		"mean_evec":[0.872,0.893,0.850,0.938,0.905,0.916]
	}

	labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	data = {
		"gru-all-best":[0.760,0.940,0.697,0.916,0.926,0.775],#6
		"gru-all-worst":[0.915,0.937,0.75,0.910,0.907,0.864],
		"mean_evec":[0.872,0.893,0.850,0.938,0.905,0.916]
	}
	x = np.arange(len(labels))  # the label locations
	width = 0.25  # the width of the bars

	print(len(list(data)))
	
	
	fig, ax = plt.subplots(figsize=(8,5),dpi=150)
	bar_name = list(data.keys())
	bar_value = list(data.values())

	print(bar_name)
	print(bar_value)

 

	rects1 = ax.bar(x - width, bar_value[0], width, label=bar_name[0])
	rects2 = ax.bar(x, bar_value[1], width, label=bar_name[1])
	rects3 = ax.bar(x + width, bar_value[2], width, label=bar_name[2])
	# rects4 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# rects1 = ax.bar(x - width, bar_value[2], width, label=bar_name[2])
	# rects2 = ax.bar(x, bar_value[4], width, label=bar_name[4])
	# rects3 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Testing AUROC')
	ax.set_title('100kb GRU')
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	plt.yticks(fontsize=10)
	ax.legend(loc='upper left')

	plt.ylim([0.65,1.0])

	ax.bar_label(rects1, padding=3, fontsize=8)
	ax.bar_label(rects2, padding=3, fontsize=8)
	ax.bar_label(rects3, padding=3, fontsize=8)
	# ax.bar_label(rects4, padding=3)

	fig.tight_layout()

	plt.show()
	plt.savefig(img_path)
	plt.clf()

def plotBarGraph_four_bar_noNHEK_HUVEC(exp_dpath, labels = None, data = None):
	img_path = os.path.join(exp_dpath,"100_50kb_cnn.png")
	
	labels = ["IMR90","K562", "HMEC", "GM12878"]
	data = {
		"100kb_5cells":[0.892,0.898,0.904,0.871], #0
		"100kb_mean":[0.877,0.881,0.946,0.939],
		"100kb_single":[0.923,0.912,0.904,0.901], 
		"100kb_all_mean":[0.872,0.893,0.938,0.905],#3
		"100kb_all":[0.908,0.939,0.919,0.911],
		"100kb_4cells":[0.911,0.936,0.929,0.912],#5
		"100kb_4cells_mean":[0.867,0.896,0.913,0.910],
		"100kb_3cells":[0.813,0.931,0.884,0.918],
		"100kb_3cells_mean":[0.834,0.889,0.903,0.906]
	}

	x = np.arange(len(labels))  # the label locations
	width = 0.2  # the width of the bars

	print(len(list(data)))
	
	
	fig, ax = plt.subplots(figsize=(8,5))
	bar_name = list(data.keys())
	bar_value = list(data.values())

	print(bar_name)
	print(bar_value)

 

	rects1 = ax.bar(x - 1.5*width, bar_value[5], width, label=bar_name[5])
	rects2 = ax.bar(x - 0.5 * width, bar_value[6], width, label=bar_name[6])
	rects3 = ax.bar(x + 0.5 * width, bar_value[7], width, label=bar_name[7])
	rects4 = ax.bar(x + 1.5*width, bar_value[8], width, label=bar_name[8])

	# rects1 = ax.bar(x - width, bar_value[2], width, label=bar_name[2])
	# rects2 = ax.bar(x, bar_value[4], width, label=bar_name[4])
	# rects3 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Testing AUROC')
	ax.set_title('100kb all chromosome, GRU vs Mean Eigenvectors')
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	plt.yticks(fontsize=10)
	ax.legend(loc='upper left')

	plt.ylim([0.7,1.0])

	ax.bar_label(rects1, padding=3,fontsize=8)
	ax.bar_label(rects2, padding=3,fontsize=8)
	ax.bar_label(rects3, padding=3,fontsize=8)
	ax.bar_label(rects4, padding=3,fontsize=8)

	fig.tight_layout()

	plt.show()
	plt.savefig(img_path)
	plt.clf()

def plotBarGraph_sive_bar_noNHEK_HUVEC(exp_dpath, labels = None, data = None):
	img_path = os.path.join(exp_dpath,"100_50kb_cnn.png")
	
	labels = ["IMR90","K562", "HMEC", "GM12878"]
	data = {
		"100kb_5cells":[0.892,0.898,0.904,0.871], #0
		"100kb_mean":[0.877,0.881,0.946,0.939],
		"100kb_single":[0.923,0.912,0.904,0.901], 
		"100kb_all_mean":[0.872,0.893,0.938,0.905],#3
		"100kb_all":[0.908,0.939,0.919,0.911],
		"100kb_4cells":[0.911,0.936,0.929,0.912],#5
		"100kb_4cells_mean":[0.867,0.896,0.913,0.910],
		"100kb_3cells":[0.813,0.931,0.884,0.918],
		"100kb_3cells_mean":[0.834,0.889,0.903,0.906]
	}

	x = np.arange(len(labels))  # the label locations
	width = 0.2  # the width of the bars

	print(len(list(data)))
	
	
	fig, ax = plt.subplots(figsize=(8,5))
	bar_name = list(data.keys())
	bar_value = list(data.values())

	print(bar_name)
	print(bar_value)

 

	rects1 = ax.bar(x - 1.5*width, bar_value[5], width, label=bar_name[5])
	rects2 = ax.bar(x - 0.5 * width, bar_value[6], width, label=bar_name[6])
	rects3 = ax.bar(x + 0.5 * width, bar_value[7], width, label=bar_name[7])
	rects4 = ax.bar(x + 1.5*width, bar_value[8], width, label=bar_name[8])

	# rects1 = ax.bar(x - width, bar_value[2], width, label=bar_name[2])
	# rects2 = ax.bar(x, bar_value[4], width, label=bar_name[4])
	# rects3 = ax.bar(x + width, bar_value[3], width, label=bar_name[3])

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Testing AUROC')
	ax.set_title('100kb all chromosome, GRU vs Mean Eigenvectors')
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	plt.yticks(fontsize=10)
	ax.legend(loc='upper left')

	plt.ylim([0.7,1.0])

	ax.bar_label(rects1, padding=3,fontsize=8)
	ax.bar_label(rects2, padding=3,fontsize=8)
	ax.bar_label(rects3, padding=3,fontsize=8)
	ax.bar_label(rects4, padding=3,fontsize=8)

	fig.tight_layout()

	plt.show()
	plt.savefig(img_path)
	plt.clf()


if __name__ == "__main__":
    main()