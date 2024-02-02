import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from numpy import random
import os
from DataLoader import dataLoader
import pandas as pd
import sys

def load_principal_components(file_path="PrincipalComponents/PCs.txt"):
	PCs = np.loadtxt(file_path, skiprows=1, unpack=True)
	return PCs

def load_training_file_names(file_path="metadata/TrainingFileNames.txt"):
	with open(file_path, 'r') as file:
		content = file.read()
	return np.array([name for name in content.split('\n') if name])

def load_testing_data(path, num_time_bins, excluded_names):
	IMG, cell_names = dataLoader(path, cutoff=num_time_bins, notTheseNames=excluded_names)
	return np.array(IMG), np.array(cell_names)

def save_testing_file_names(names, filename="metadata/TestingFileNames.txt"):
	with open(filename, 'w') as file:
		for name in names:
			file.write("{}\n".format(name))

def process_images_for_projection(IMG, PCsI, num_time_bins):
	column_height = 256 * 256
	all_img_projs = []

	for imgs in tqdm(IMG):
		X = np.zeros((column_height, num_time_bins))
		for i, img in enumerate(imgs):
			X[:, i] = img.flatten()

		NEW = PCsI @ X.T
		img_projs = np.reshape(NEW, (1, 256, 256))

		all_img_projs.append(img_projs)

	return np.array(all_img_projs)

def display_random_projections(all_img_projs, num_images, num_PCs):
	rand_image_indices = np.random.choice(np.arange(0, num_images), size=4, replace=False)

	fig = plt.figure(figsize=(10, 8))

	for i, rand_indx in enumerate(rand_image_indices):
		fig.add_subplot(2, 2, i + 1)
		plt.imshow(all_img_projs[rand_indx][0], cmap="inferno")
		plt.title("Indx {} PCs {}".format(rand_indx, num_PCs))
		plt.colorbar()

	plt.tight_layout()
	plt.savefig("figures/projections/PC1_imgRecon.png")

def main():
	PCs = load_principal_components()
	PCsI = np.sqrt(PCs[0]**2 + PCs[1]**2 + PCs[2]**2)

	num_PCs, length = np.shape(PCs)
	print("Number of PC(s) used for reconstruction: {}".format(num_PCs))

	training_file_names = load_training_file_names()

	path2Exp = '../EIT-System/FLIM-ISS_Time-decay'
	IMG, cell_names = load_testing_data(path2Exp, num_time_bins=length, excluded_names=training_file_names)

	save_testing_file_names(cell_names)

	all_img_projs = process_images_for_projection(IMG, PCsI, num_time_bins=length)

	num_images, PC, height, width = np.shape(all_img_projs)

	display_random_projections(all_img_projs, num_images, num_PCs)

if __name__ == "__main__":
	main()


