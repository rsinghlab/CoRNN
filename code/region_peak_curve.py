import matplotlib.pyplot as plt

import numpy as np

import datetime
import os

def main():

	now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	exp_dpath = os.path.join("data","exp_data","combined_curve","exp_{}".format(now))
	if not os.path.exists(exp_dpath):
		os.makedirs(exp_dpath)

	print(exp_dpath)

	# X = [0,1,2,3,4,5]
	# Y = [{"cell":"IMR90", "curve":"model", "label":"IMR90-model",
	# 		"data":[0.630,0.627,0.691,0.792,0.830,0.834]},
	# 	]
	# img_path = os.path.join(exp_dpath,"IMR90.png")
	# title = 'IMR90 - AUROC score by region'
	# generate_curve(X,Y,img_path,title)

	# X = [0,1,2,3,4,5]
	# Y = [{"cell":"GM12878", "curve":"model", "label":"GM12878-model",
	# 		"data":[0.857,0.851,0.832,0.836,0.815,0.827]},
	# 	]
	# img_path = os.path.join(exp_dpath,"GM12878.png")
	# title = 'GM12878 - AUROC score by region'
	# generate_curve(X,Y,img_path,title)

	# X = [0,1,2,3,4,5]
	# Y = [{"cell":"HMEC", "curve":"model", "label":"HMEC-model",
	# 		"data":[0.849,0.831,0.804,0.795,0.728,0.822]},
	# 	]
	# img_path = os.path.join(exp_dpath,"HMEC.png")
	# title = 'HMEC - AUROC score by region'
	# generate_curve(X,Y,img_path,title)

	# X = [0,1,2,3,4,5]
	# Y = [{"cell":"HUVEC", "curve":"model", "label":"HUVEC-model",
	# 		"data":[0.890,0.726,0.764,0.772,0.667,0.844]},
	# 	]
	# img_path = os.path.join(exp_dpath,"HUVEC.png")
	# title = 'HUVEC - AUROC score by region'
	# generate_curve(X,Y,img_path,title)

	# X = [0,1,2,3,4,5]
	# Y = [{"cell":"K562", "curve":"model", "label":"K562-model",
	# 		"data":[0.898,0.885,0.863,0.853,0.845,0.792]},
	# 	]
	# img_path = os.path.join(exp_dpath,"K562.png")
	# title = 'K562 - AUROC score by region'
	# generate_curve(X,Y,img_path,title)

	# X = [0,1,2,3,4,5]
	# Y = [{"cell":"NHEK", "curve":"model", "label":"NHEK-model",
	# 		"data":[0.566,0.594,0.583,0.574,0.486,0.536]},
	# 	]
	# img_path = os.path.join(exp_dpath,"NHEK.png")
	# title = 'NHEK - AUROC score by region'
	# generate_curve(X,Y,img_path,title)


	# X = [0,1,2,3,4,5]
	# Y = [{"cell":"IMR90", "curve":"model", "label":"IMR90-model",
	# 		"data":[0.630,0.627,0.691,0.792,0.830,0.834]},
	# 	{"cell":"GM12878", "curve":"model", "label":"GM12878-model",
	# 		"data":[0.857,0.851,0.832,0.836,0.815,0.827]},	
	# 	{"cell":"HMEC", "curve":"model", "label":"HMEC-model",
	# 		"data":[0.849,0.831,0.804,0.795,0.728,0.822]},
	# 	{"cell":"HUVEC", "curve":"model", "label":"HUVEC-model",
	# 		"data":[0.890,0.726,0.764,0.772,0.667,0.844]},
	# 	{"cell":"K562", "curve":"model", "label":"K562-model",
	# 		"data":[0.898,0.885,0.863,0.853,0.845,0.792]},
	# 	{"cell":"NHEK", "curve":"model", "label":"NHEK-model",
	# 		"data":[0.566,0.594,0.583,0.574,0.486,0.536]},	
	# 	]
	# img_path = os.path.join(exp_dpath,"all.png")
	# title = 'AUROC score by region - 100kb GRU (add mean)'
	# generate_curve(X,Y,img_path,title)

	X = [0,1,2,3,4,5]
	Y = [{"cell":"IMR90", "curve":"model", "label":"IMR90-model",
			"data":[0.741,0.689,0.701,0.758,0.767,0.747]},
		]
	img_path = os.path.join(exp_dpath,"IMR90.png")
	title = 'IMR90 - AUROC score by region'
	generate_curve(X,Y,img_path,title)

	X = [0,1,2,3,4,5]
	Y = [{"cell":"GM12878", "curve":"model", "label":"GM12878-model",
			"data":[0.874,0.865,0.853,0.854,0.837,0.858]},
		]
	img_path = os.path.join(exp_dpath,"GM12878.png")
	title = 'GM12878 - AUROC score by region'
	generate_curve(X,Y,img_path,title)

	X = [0,1,2,3,4,5]
	Y = [{"cell":"HMEC", "curve":"model", "label":"HMEC-model",
			"data":[0.792,0.791,0.774,0.747,0.689,0.769]},
		]
	img_path = os.path.join(exp_dpath,"HMEC.png")
	title = 'HMEC - AUROC score by region'
	generate_curve(X,Y,img_path,title)

	X = [0,1,2,3,4,5]
	Y = [{"cell":"HUVEC", "curve":"model", "label":"HUVEC-model",
			"data":[0.643,0.589,0.594,0.602,0.555,0.691]},
		]
	img_path = os.path.join(exp_dpath,"HUVEC.png")
	title = 'HUVEC - AUROC score by region'
	generate_curve(X,Y,img_path,title)

	X = [0,1,2,3,4,5]
	Y = [{"cell":"K562", "curve":"model", "label":"K562-model",
			"data":[0.864,0.870,0.846,0.842,0.835,0.761]},
		]
	img_path = os.path.join(exp_dpath,"K562.png")
	title = 'K562 - AUROC score by region'
	generate_curve(X,Y,img_path,title)

	X = [0,1,2,3,4,5]
	Y = [{"cell":"NHEK", "curve":"model", "label":"NHEK-model",
			"data":[0.601,0.619,0.613,0.645,0.529,0.582]},
		]
	img_path = os.path.join(exp_dpath,"NHEK.png")
	title = 'NHEK - AUROC score by region'
	generate_curve(X,Y,img_path,title)


	X = [0,1,2,3,4,5]
	Y = [{"cell":"IMR90", "curve":"model", "label":"IMR90-model",
			"data":[0.741,0.689,0.701,0.758,0.767,0.747]},
		{"cell":"GM12878", "curve":"model", "label":"GM12878-model",
			"data":[0.874,0.865,0.853,0.854,0.837,0.858]},	
		{"cell":"HMEC", "curve":"model", "label":"HMEC-model",
			"data":[0.792,0.791,0.774,0.747,0.689,0.769]},
		{"cell":"HUVEC", "curve":"model", "label":"HUVEC-model",
			"data":[0.643,0.589,0.594,0.602,0.555,0.691]},
		{"cell":"K562", "curve":"model", "label":"K562-model",
			"data":[0.864,0.870,0.846,0.842,0.835,0.761]},
		{"cell":"NHEK", "curve":"model", "label":"NHEK-model",
			"data":[0.601,0.619,0.613,0.645,0.529,0.582]},	
		]
	img_path = os.path.join(exp_dpath,"all.png")
	title = 'AUROC score by region - 100kb GRU (add mean)'
	generate_curve(X,Y,img_path,title)

	

	

def generate_curve(X,Y,img_path,title):

	fig, ax = plt.subplots(figsize=(8, 6))
	ax.set_title(title, fontsize=15)

	for curve in Y:
		ax.plot(X,curve["data"],label = curve["label"])
	ax.set_ylabel('Testing AUROC')
	ax.set_xlabel('Region')
	plt.legend(loc="best")
	plt.savefig(img_path)
	plt.clf()

	# plt.show()



if __name__ == "__main__":
    main()