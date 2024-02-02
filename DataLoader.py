# my_module.py

__version__ = "1.1.0"

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os
import time


def read_tiff(path, cutoff=None, only_odds=False, only_evens=False):
	"""Load direct tiff files.

	Args:
		path (str): Path to the location of experiment files (Exp1, Exp2,...).
		cutoff (int): Cutoff for the number of frames to read from each cell.
		only_odds (bool): Whether to read only odd frames.
		only_evens (bool): Whether to read only even frames.

	Returns:
		list: A list of image arrays.
	"""

	img = Image.open(path)

	if cutoff is not None:
		trunc = img.n_frames - cutoff
	else:
		trunc = 0

	step = 2 if only_evens or only_odds else 1
	start = 1 if only_odds else 0

	images = []
	for i in tqdm(range(start, img.n_frames - trunc, step), desc="Loading images"):
		img.seek(i)
		img2array = np.array(img)
		images.append(img2array)

	return images


def data_loader(path2exp, num_cells_per=1, cutoff=None, only_odds=False, only_evens=False, not_these_names=None):
	"""Load data from experiment files.

	Args:
		path2exp (str): Path to the location of experiment files (Exp1, Exp2,...).
		num_cells_per (int): Number of cells taken from each file within the Exp files.
		cutoff (int): Cutoff for the number of frames to read from each cell.
		only_odds (bool): Whether to read only odd frames.
		only_evens (bool): Whether to read only even frames.
		not_these_names (list): List of cell names to exclude.

	Returns:
		tuple: A tuple containing a list of image arrays and a list of cell names.
	"""
	start_time = time.time()

	def remove_ds_store(files):
		if '.DS_Store' in files:
			files.remove('.DS_Store')

	exp_files = sorted(os.listdir(path2exp))  # gets me to Exp files
	remove_ds_store(exp_files)

	images_list = []
	cell_names = []
	num_cells = 0

	for exp in exp_files:
		sub_exp = sorted(os.listdir(os.path.join(path2exp, exp)))  # gets me to subExps
		remove_ds_store(sub_exp)

		for se in sub_exp:
			cell_folders = sorted(os.listdir(os.path.join(path2exp, exp, se)))
			remove_ds_store(cell_folders)

			for cf in cell_folders:
				cells = sorted(os.listdir(os.path.join(path2exp, exp, se, cf)))
				remove_ds_store(cells)
				random.shuffle(cells)

				for names in cells[:num_cells_per]:
					if not_these_names is None or names not in not_these_names:
						cell_names.append(names)
						img = read_tiff(os.path.join(path2exp, exp, se, cf, names), cutoff=cutoff,
                                        only_odds=only_odds, only_evens=only_evens)
						images_list.append(img)
						num_cells += 1

	print("Total number of cells: {}".format(num_cells))
    
	end_time = time.time()
	elapsed_time = end_time - start_time
	print("Load time: {:.3} mins\n".format(elapsed_time / 60))

	return images_list, cell_names


def main():
	start_time = time.time()

	path_to_exp = '../EIT-System/FLIM-ISS_Time-decay'
	imgs, names = data_loader(path_to_exp, num_cells_per=6, cutoff=53, only_odds=True)

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
	print("Total run time: {.3} mins\n".format(elapsed_time / 60))


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

