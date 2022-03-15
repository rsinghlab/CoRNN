
import matplotlib.pyplot as plt

import numpy as np

import datetime
import os
import csv



from matplotlib import cm
# from colorspacious import cspace_converter
from collections import OrderedDict
from matplotlib import colors as mcolors

def main():


	now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	exp_dpath = os.path.join("data","exp_data","combined_curve","exp_{}".format(now))
	if not os.path.exists(exp_dpath):
		os.makedirs(exp_dpath)

	print(exp_dpath)

	# data_load_path = os.path.join("data","Testing_Results","perturbation")
	data_load_path = os.path.join("data","100kb_cell_specific_regions","test_results")

	# data_load_path = os.path.join("data","exp_data","combined_curve","exp_2021-09-14-16-22-31")

	# exp_2021-09-15-00-09-10 -GRU without mean, & exp_2021-09-15-09-39-28

	cell_lines = ["K562", "NHEK", "HMEC", "GM12878", "HUVEC","IMR90"]
	# cell_lines = ["K562"]
	for target_cell in cell_lines:
		file_path = os.path.join(data_load_path,"{}_perturbation.csv".format(target_cell))
		hm = []
		labels = []
		auroc = []
		# accu =[]
		with open(file_path, newline='') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			for line_count, row in enumerate(reader):
				if line_count == 0:
					hm = row[1:7]
				else:
					labels.append(row[1:7])
					auroc.append(float(row[7]))
					# accu.append(float(row[8]))
		
		print(auroc)
		img_path = os.path.join(exp_dpath,"100kb_perturbation_{}.jpg".format(target_cell))
		bar_width = 0.5  # the width of the bar
		fig, ax = plt.subplots(figsize=(15,5),dpi=100)
		data = np.array(auroc)
		# data = np.array(accu)
		labels = np.array(labels)

		sort_index = np.argsort(data)

		sorted_data = []
		sorted_labels = []

		for idx in reversed(sort_index):
			sorted_data.append(data[idx])
			sorted_labels.append(labels[idx])


		index = np.arange(len(data))
		# ax.grid(zorder=0)
		bars = []
		for idx, score in enumerate(sorted_data):
			bar = plt.bar(index[idx],score,bar_width)
			bars.append(bar)
		# plt.bar(index, auroc, bar_width)
		sorted_labels = np.array(sorted_labels) #63 *6
		sorted_labels = np.transpose(sorted_labels) #6*63

		#find where is the mean baseline
		mean_idx = 0

		for i in range(len(sorted_labels[0])):
			set_list = set(sorted_labels[:,i])
			set_list = list(set_list)
			# print(sorted_labels[:,i])
			if len(set_list) == 1 and set_list[0] == '1':
				print(i)
				print(sorted_labels[:,i])
				mean_idx = i

		text_row = []
		text = []

		for i in range(len(sorted_labels[0])):
			text_row.append("")
		
		for i in range(len(sorted_labels)):
			text.append(text_row)


		the_table = plt.table(cellText=text,
			fontsize=3,
           rowLabels=hm,
           loc='bottom',
           bbox=[0.043, -0.32, 0.915, 0.3]
          )
		for i in range(len(sorted_labels)):
			for j in range(len(sorted_labels[0])):
				flag = int(sorted_labels[i,j])
				if flag == 0:
					the_table[(i, j)].set_facecolor("#56b5fd")
		
		for key, cell in the_table.get_celld().items():
			cell.set_linewidth(0.5)
		
		plt.tick_params(
    		axis='x',          # changes apply to the x-axis
    		which='both',      # both major and minor ticks are affected
    		bottom=False,      # ticks along the bottom edge are off
   			top=False,         # ticks along the top edge are off
    		labelbottom=False)

		for idx, bar in enumerate(bars):
			ax.bar_label(bar, padding=3, fontsize=4,fmt='%.3f')
			# if idx == mean_idx:
			# 	ax.bar_label(bar, padding=3, fontsize=4,fmt='%.3f',color='red',weight='bold')
				# ax.text(idx-0.5, sorted_data[i]-0.01, 'Mean',fontsize=5,
    #     				bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 2})

			# else:
				
			# 	ax.bar_label(bar, padding=3, fontsize=4,fmt='%.3f')


	
		ax.set_ylabel('Testing AUROC')
		ax.set_title("{} perturbation test".format(target_cell))

		# ax.set_xticks(index)
		# ax.set_xticklabels(labels)
		# plt.yticks(fontsize=5)
		# ax.legend(loc='lower left', fontsize = 8)

		plt.ylim([0.0,0.2])


		fig.tight_layout()

		# plt.show()
		plt.savefig(img_path,dpi=200)
		plt.clf()








if __name__ == "__main__":
    main()
