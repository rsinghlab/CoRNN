import os
import pandas as pd
import pyBigWig
import numpy as np
from scipy.stats import pearsonr

def calculate_correlation(cell):
	bigwig_file = os.path.join("data","NAWG_workspace","bigwig","K562","K562_H3K4me1_ENCFF900IMQ.bigWig")
	bw1 = pyBigWig.open(bigwig_file)
	bw1dict = bw1.chroms()
	keys = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
        'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 
        'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22']
	correlation = {}
	save_dir = os.path.join("data","correlation_workspace",cell)
	if not os.path.isdir(save_dir):
			os.mkdir(save_dir)
	for key in keys:
		arr = np.array([['mean']])
		chrom = key
		start = 0
		basepairs = bw1.chroms()[chrom]
		res = 100000

		while (start + res) < basepairs:
			mean = bw1.stats(chrom, start, start + res, type = "mean")#using the built in function to calculate mean 
			line = [mean]
			#arr = np.vstack([arr, line])
			arr = np.vstack((arr, line))
			start += res

		
		if not os.path.isdir(save_dir):
			os.mkdir(save_dir)
		
		np.savetxt(os.path.join(save_dir,'{}_H3K4me1_{}_100kb.txt'.format(cell,chrom)), np.array(arr), fmt="%s")

	
	
	for i in range(1, 23): 
		bigwig_f = open(os.path.join(save_dir,'{}_H3K4me1_chr{}_100kb.txt'.format(cell,i)), "r")
		bigwig = bigwig_f.readlines()

		eigen_f = open(os.path.join("data","NAWG_workspace","Compartments",cell,"100kb",\
			"{}_eigen_100kb_chr{}.txt".format(cell,i)), "r")
		eigen = eigen_f.readlines()

		bigwig_data = [item.strip() for item in bigwig]
		eigen_data = [item.strip() for item in eigen]

		#remove the first line of bigwig data - header
		#remove the last window of eigenvector - to match the length
		bigwig_data = bigwig_data[1:]
		eigen_data = eigen_data[:-1]
		
		new_bigwig_data = []
		new_eigen_data = []

		print(len(bigwig_data))
		print(len(eigen_data))

		for (b,e) in zip(bigwig_data,eigen_data):

			if b == "NaN" or b == "None" or e == "NaN" or e == "None":
				continue
			else:
				new_bigwig_data.append(float(b))
				new_eigen_data.append(float(e))


		print(len(new_bigwig_data))
		print(len(new_eigen_data))
		corr, _ = pearsonr(new_bigwig_data, new_eigen_data)
		print(corr)

		# print()
		# data = pd.concat([bigwig_df, eigen_df], axis = 1)
		# data = data.replace('None', np.nan).dropna()
		# correlation_df = data.corr()
		# print(correlation_df)
		# correlations = correlation_df["mean"]
		# corr = np.array(correlations)[1] 
		correlation["chr{}".format(i)] = corr
		#print(corr)
		# if corr < 0:
		# 	new_eigen = eigen_df.abs()
		# 	new_eigen.to_csv(f'K562_eigen_100kb_chr{i}.csv') 
		# else: 
		# 	eigen_df.to_csv(f'K562_eigen_100kb_chr{i}.csv')

	return correlation



# corre  = calculate_correlation("K562")
# print(corre)