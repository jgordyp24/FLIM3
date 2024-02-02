import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import animation
from PIL import Image
import os
import itertools
import threading
import time
import sys


class cumulativePCA:
	def __init__(self, images):
		self.images = images
		self.numImgs, self.ydim, self.xdim = np.shape(images)
		self.singular_values = None


	def removeDCoffset(self, k_size, N):
		'''k_size	--	tuple (x,y) Kernal idecies
			N 	--	integer number of last images used in DC noise reduction'''

		print("Reducing DC offset")
		k_x = np.arange(k_size[0])
		k_y = np.arange(k_size[1])

		lastIMGs = self.images[-N:]

		I = []
		for img in lastIMGs:
			for ky in k_y:
				for kx in k_x:
					I.append(img[ky][kx])
		
		DC_off = np.average(I)

		DC_corrected = []
		for img in self.images:
			DC_corrected.append(img - 10+0*DC_off)

		self.images = DC_corrected

		return None


	#Noise correction procedure
	def NoiseCorrection(self):
		'''Noise-Correction Procedure (NCP) Each 
		image is divided by the square root of its 
		average intensity
		'''

		print("Reducing Poisson's noise")

		NCIs = [] # list for noise corrected images
		for img in self.images:
			avg = np.mean(img)
			NCIs.append(img/np.sqrt(avg))

		self.images = NCIs

		return None

	def getPCs(self, PCs):

		# rowHeight = self.ydim * self.xdim
		# X = np.zeros((self.numImgs, rowHeight))
		
		print("Extracting {} Principle Component(s)".format(PCs))

		columnHeight = self.ydim * self.xdim


		X = np.zeros((columnHeight, self.numImgs))
		for i, img in tqdm(enumerate(self.images), total=len(self.images), desc="Shaping data matrix"):
			X[:,i] = img.flatten()
		# for i, img in enumerate(self.images):
		# 	X[i,:] = img.flatten()		

		print("\nData matrix shape:", np.shape(X))

		print("Standardization in progress...")
		std_scaler = StandardScaler()
		scaled_df = std_scaler.fit_transform(X)
		print("Standardization complete.")

		print("PCA in progress...")
		pca = PCA(n_components=PCs)
		pca.fit_transform(scaled_df)
		print("PCA complete.")

		print("Access singular values...")
		self.singular_values = pca.singular_values_
		print("Done.")


		return pca.components_, pca.mean_


	# def getPCs(self, PCs):
		
	# 	rowHeight = self.ydim * self.xdim
	# 	X = np.zeros((self.numImgs, rowHeight))
	# 	print("Shaping data matrix")
	# 	for i, img in enumerate(self.images):
	# 		X[i,:] = img.flatten()

	# 	Xavg = np.mean(X, axis=1)

	# 	B = X #- np.tile(Xavg, (self.ydim * self.xdim, 1)).T

	# 	print("Running PCA on images...")
	# 	pca = PCA(n_components=PCs)
	# 	pca.fit(B)

	# 	components = pca.fit_transform(B)
	# 	# image_recon = pca.inverse_transform(components)

	# 	return components


	def decomposition(self):
		'''We just doing SVD which is the bedrock of PCA'''

		columnHeight = self.ydim * self.xdim

		X = np.zeros((columnHeight, self.numImgs))
		for i, img in enumerate(self.images):
			X[:,i] = img.flatten()

		print("Performing decomposition...")

		U, S, VT = LA.svd(X, full_matrices=False)
		S_diag = np.diag(S)

		print("U, Sigma, V.T matrices calculated.")

		return U, S_diag, VT


	def get_singular_vals(self):
		return self.singular_values

	def get_singular_vals_plot(self, Sigma, r=None):
		S = np.diag(Sigma)

		plt.figure()
		plt.semilogy(S)
		if r!=None:
			plt.semilogy(S[:r], c="red")
		plt.title("Singular Values")
		plt.xlabel("PCs")
		plt.show()


	def PC_ExplainedVar_plot(self, Sigma, r=None):
		S = np.diag(Sigma)

		plt.figure(figsize=(10, 4))
		plt.plot(np.cumsum(S)/np.sum(S)*100)
		if r!=None:
			plt.plot(np.cumsum(S[:r])/np.sum(S[:r])*100)
		plt.title("Singular Values: Cumulative Sum")
		plt.xlabel("PC")
		plt.ylabel("Varience")
		plt.show()

