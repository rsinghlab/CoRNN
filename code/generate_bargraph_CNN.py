import matplotlib.pyplot as plt
import numpy as np
import os

def main():
	auroc_data = {
		"Mean Baseline":[0.872,0.893,0.850,0.938,0.905,0.916],
		"CoRNN":[0.901,0.938,0.857,0.957,0.953,0.916],
		"CNN": [0.892,0.898,0.818,0.904,0.871,0.865]
	}


	labels = ["IMR90","K562", "NHEK", "HMEC", "GM12878", "HUVEC"]
	title = "Testing Performance"
	# colors = ["tab:blue","tab:orange","tab:cyan","tab:purple"]
	plotBarGraph(labels,auroc_data,title)


def plotBarGraph(labels, data, title):
	labels_order = ["GM12878", "K562", "IMR90","HMEC", "NHEK", "HUVEC"]
	
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
	bar_val_updated = np.zeros((3,6))
	
	for i, label in enumerate(labels_order):
		bar_val_updated[:,i] = val_by_cell[label]
		
	print(bar_val_updated)
	bar_value = bar_val_updated


	fig, ax = plt.subplots(figsize=(12, 8))
	x = np.arange(len(labels_order))

	# Define bar width. We'll use this to offset the second bar.
	bar_width = 0.2

	# Note we add the `width` parameter now which sets the width of each bar.
	b1 = ax.bar(x-bar_width, bar_value[0],label=bar_name[0],width=bar_width)
	b2 = ax.bar(x, bar_value[1] ,label=bar_name[1],width=bar_width)
	b3 = ax.bar(x + bar_width, bar_value[2],label=bar_name[2], width=bar_width)
	plt.ylim([0.7,1.0])
	plt.title("Testing Performance (CoRNN vs CNN)")
	ax.set_ylabel('Testing AUROC')

	ax.set_xticks(x + bar_width / 2)
	ax.set_xticklabels(labels_order)
	ax.legend(loc='lower left',fontsize = 8)


	ax.bar_label(b1, padding=3, fontsize=8)
	ax.bar_label(b2, padding=3, fontsize=8)
	ax.bar_label(b3, padding=3, fontsize=8)

	plt.show()
	plt.savefig("CoRNN-CNN.eps", format='eps')

if __name__ == "__main__":
    main()