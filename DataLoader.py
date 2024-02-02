import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os


def read_tiff(path, cutoff=None, onlyOdds=False, onlyEvens=False):
	"""Load direct tiff files.

	Args:
		path (str): Path to the location of experiment files (Exp1, Exp2,...).
		cutoff (int): Cutoff for the number of frames to read from each cell.
		onlyOdds (bool): Whether to read only odd frames.
		onlyEvens (bool): Whether to read only even frames.

	Returns:
		list: A list of image arrays.
	"""

	img = Image.open(path)
    
	if cutoff is not None:
		trunc = img.n_frames - cutoff
	else:
		trunc = 0

	step = 2 if onlyEvens or onlyOdds else 1
	start = 1 if onlyOdds else 0

	images = []
	for i in tqdm(range(start, img.n_frames - trunc, step), desc="Loading images"):
		img.seek(i)
		img2array = np.array(img)
		images.append(img2array)

	return images



def dataLoader(path2Exp, numCellsPer=1, cutoff=None, onlyOdds=False, onlyEvens=False, notTheseNames=None):
	"""Load data from experiment files.

	Args:
		path2Exp (str): Path to the location of experiment files (Exp1, Exp2,...).
		numCellsPer (int): Number of cells taken from each file within the Exp files.
		cutoff (int): Cutoff for the number of frames to read from each cell.
		onlyOdds (bool): Whether to read only odd frames.
		onlyEvens (bool): Whether to read only even frames.
		notTheseNames (list): List of cell names to exclude.

	Returns:
		tuple: A tuple containing a list of image arrays and a list of cell names.
	"""

	def remove_DS_store(files):
		if '.DS_Store' in files:
			files.remove('.DS_Store')

	ExpFiles = sorted(os.listdir(path2Exp))  # gets me to Exp files
	remove_DS_store(ExpFiles)

	ImagesList = []
	cellNames = []
	numCells = 0

	for exp in ExpFiles:
		subExp = sorted(os.listdir(os.path.join(path2Exp, exp)))  # gets me to subExps
		remove_DS_store(subExp)

		for SE in subExp:
			cellFolders = sorted(os.listdir(os.path.join(path2Exp, exp, SE)))
			remove_DS_store(cellFolders)

			for CF in cellFolders:
				cells = sorted(os.listdir(os.path.join(path2Exp, exp, SE, CF)))
				remove_DS_store(cells)
				random.shuffle(cells)

				for names in cells[:numCellsPer]:
					if notTheseNames is None or names not in notTheseNames:
						cellNames.append(names)
						img = read_tiff(os.path.join(path2Exp, exp, SE, CF, names), cutoff=cutoff,
                                        onlyOdds=onlyOdds, onlyEvens=onlyEvens)
						ImagesList.append(img)
						numCells += 1

	print("Total number of cells: {}".format(numCells))

	return ImagesList, cellNames
