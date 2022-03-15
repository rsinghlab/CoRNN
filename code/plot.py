# importing the required module
import matplotlib.pyplot as plt
import csv
import numpy as np


values=[]
index = []

count=0

with open('data/gm12878_chr_1_test.csv', newline='') as csvfile:
    compartments = csv.reader(csvfile, delimiter=',')
    for row in compartments:
        if count>0:
            values.append(float(row[8]))
            index.append(count)
        count=count+1
 

# frequencies
x = np.array(index)
y = np.array(values)


plt.figure(figsize=(2,20))

colors = np.array([(216/255, 177/255, 1)]*len(y))
colors[y >= 0] = (1, 102/255, 0)
plt.barh(x,y,color = colors, linewidth=0)
plt.axis('off')

# function to show the plot
plt.savefig("gm12878.jpeg", bbox_inches = 'tight')
#plt.show()