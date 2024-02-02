import numpy as np
import matplotlib.pyplot as plt
# from DataLoader import dataLoader
from tqdm import tqdm
from PIL import Image, ImageSequence
from numpy import random
import os
import functools
import cumulativePCA
import time
import sys
# sys.path.append('../../General Scripts')
from DataLoader import dataLoader


start_time = time.time()
#
# path2Exp = '../TrainingOld/CellLine0'
# path2Exp = '../TrainingNew/CellLine0'

path2Exp = '../EIT-System/FLIM-ISS_Time-decay'

IMGs, Names = dataLoader(path2Exp, numCellsPer=6, cutoff=53, onlyEvens=True)

nameFile = open("metadata/TrainingFileNames.txt", 'w')
for names in Names:
	nameFile.write("{}\n".format(names))
nameFile.close()

IMG = np.array(IMGs)

# print(np.shape(IMG))

IMG_reorg = IMG.transpose(1,0,2,3) #reorganizes data so that the time domain is the 0th index

#Combine dimensions
new_shape = (IMG_reorg.shape[0], IMG_reorg.shape[1]*IMG_reorg.shape[2], IMG_reorg.shape[3])

combined_imgs = IMG_reorg.reshape(new_shape)


pca_c = cumulativePCA.cumulativePCA(combined_imgs)

# numPCs = 4
numPCs = len(combined_imgs)
PCs, mean = pca_c.getPCs(numPCs)


PC_yDim, PC_xDim = np.shape(PCs)

PCfile = open("PrincipalComponents/PCs.txt", "w")
PCfile.write("PC1\tPC2\tPC3\tPC4\n")
for i in range(PC_xDim):
	PCfile.write("{}\t{}\t{}\t{}\n".format(PCs[0,i], PCs[1,i], PCs[2,i], PCs[3,i]))

PCfile.close()

fig = plt.figure()
fig.add_subplot(221)
plt.plot(PCs[0], label="PC1")
plt.title("PC 1")

fig.add_subplot(222)
plt.plot(PCs[1], label="PC2")
plt.title("PC 2")

fig.add_subplot(223)
plt.plot(PCs[2], label="PC3")
plt.title("PC 3")

fig.add_subplot(224)
plt.plot(PCs[3], label="PC4")
plt.title("PC 4")

plt.tight_layout()
fig.savefig("figures/PCs/PC_plots.png")


end_time = time.time()
elapsed_time = end_time - start_time

print("Total run time: {} mins\n".format(elapsed_time/60))
