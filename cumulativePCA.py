# my_module.py

__version__ = "1.1.0"

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class CumulativePCA:
	def __init__(self, images):
		self.images = images
		self.num_imgs, self.y_dim, self.x_dim = np.shape(images)
		self.singular_values = None

	def remove_dc_offset(self, k_size, N):
		print("Reducing DC offset")
		k_x = np.arange(k_size[0])
		k_y = np.arange(k_size[1])

		last_imgs = self.images[-N:]

		I = []
		for img in last_imgs:
			for ky in k_y:
				for kx in k_x:
					I.append(img[ky][kx])

		dc_off = np.average(I)

		dc_corrected = []
		for img in self.images:
			dc_corrected.append(img - 10 + 0 * dc_off)

		self.images = dc_corrected

	def noise_correction(self):
		print("Reducing Poisson's noise")
		NCIs = []  # list for noise corrected images
		for img in self.images:
			avg = np.mean(img)
			NCIs.append(img / np.sqrt(avg + 1e-8))  # Adding a small epsilon (1e-8) for stability

		self.images = NCIs
		return None


	def get_pcs(self, num_pcs):
		print(f"Extracting {num_pcs} Principal Component(s)")

		column_height = self.y_dim * self.x_dim
		X = np.zeros((column_height, self.num_imgs))

		for i, img in tqdm(enumerate(self.images), total=len(self.images), desc="Shaping data matrix"):
			X[:, i] = img.flatten()

		print("\nData matrix shape:", np.shape(X))

		print("Standardization in progress...")
		std_scaler = StandardScaler()
		scaled_df = std_scaler.fit_transform(X)

		print("Checking for NaN or infinity values...")
		print("NaNs: ", np.isnan(scaled_df).any())
		print("Infs: ", np.isinf(scaled_df).any())

		# Replace NaN or infinity values with zeros or another appropriate value
		scaled_df = np.nan_to_num(scaled_df, nan=0.0, posinf=0.0, neginf=0.0)

		print("Standardization complete.\n")

		print("PCA in progress...")
		pca = PCA(n_components=num_pcs)
		pca.fit_transform(scaled_df)
		print("PCA complete.")

		print("Access singular values...")
		self.singular_values = pca.singular_values_
		print("Done.")

		return pca.components_, pca.mean_


	def decomposition(self):
		print("Performing decomposition...")

		column_height = self.y_dim * self.x_dim
		X = np.zeros((column_height, self.num_imgs))

		for i, img in enumerate(self.images):
			X[:, i] = img.flatten()

		try:
			U, S, VT = LA.svd(X, full_matrices=False)
			S_diag = np.diag(S)
			print("U, Sigma, V.T matrices calculated.")
		except np.linalg.LinAlgError as e:
			print(f"SVD did not converge. Error: {e}")
			U, S_diag, VT = None, None, None

		return U, S_diag, VT


	def get_singular_vals(self):
		return self.singular_values

	def get_singular_vals_plot(self, Sigma, r=None):
		S = np.diag(Sigma)

		plt.figure()
		plt.semilogy(S)
		if r is not None:
			plt.semilogy(S[:r], c="red")
		plt.title("Singular Values")
		plt.xlabel("PCs")
		plt.show()

	def pc_explained_var_plot(self, Sigma, r=None):
		S = np.diag(Sigma)

		plt.figure(figsize=(10, 4))
		plt.plot(np.cumsum(S) / np.sum(S) * 100)
		if r is not None:
			plt.plot(np.cumsum(S[:r]) / np.sum(S[:r]) * 100)
		plt.title("Singular Values: Cumulative Sum")
		plt.xlabel("PC")
		plt.ylabel("Variance")
		plt.show()



