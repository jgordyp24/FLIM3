# my_module.py

__version__ = "1.1.0"

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageSequence
from DataLoader import data_loader  # Assuming data_loader is stored locally
import cumulativePCA
import time

def main():
	start_time = time.time()

	path_to_exp = '../EIT-System/FLIM-ISS_Time-decay'
	imgs, names = data_loader(path_to_exp, num_cells_per=6, cutoff=53, only_evens=True)

	with open("metadata/TrainingFileNames.txt", 'w') as name_file:
		for name in names:
			name_file.write("{}\n".format(name))

	img_array = np.array(imgs)
	img_reorg = img_array.transpose(1, 0, 2, 3)
	
	# Combine dimensions
	new_shape = (img_reorg.shape[0], img_reorg.shape[1] * img_reorg.shape[2], img_reorg.shape[3])
	combined_imgs = img_reorg.reshape(new_shape)

	pca_c = cumulativePCA.CumulativePCA(combined_imgs)

	# Use either a fixed number of PCs (e.g., 4) or the total number of images
	num_PCs = len(combined_imgs)
	PCs, mean = pca_c.get_pcs(num_PCs)

	with open("PrincipalComponents/PCs.txt", "w") as PC_file:
		PC_file.write("PC1\tPC2\tPC3\tPC4\n")
		for i in range(PCs.shape[1]):
			PC_file.write("{}\t{}\t{}\t{}\n".format(PCs[0, i], PCs[1, i], PCs[2, i], PCs[3, i]))

	plot_pcs(PCs)

	end_time = time.time()
	elapsed_time = end_time - start_time
	print("PC extraction time: {:.3} mins\n".format(elapsed_time / 60))

def plot_pcs(PCs):
	fig = plt.figure(figsize=(10, 8))
	for i in range(4):
		fig.add_subplot(2, 2, i + 1)
		plt.plot(PCs[i], label=f"PC{i + 1}")
		plt.title(f"PC {i + 1}")

	plt.tight_layout()
	fig.savefig("figures/PCs/PC_plots.png")

if __name__ == "__main__":
	main()