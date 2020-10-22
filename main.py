import numpy as np 
import sys
import os
import pdb
sys.path.append(os.getcwd() + "/TopologyComparison/")
sys.path.append(os.getcwd() + "/CreatDataset/")
from Topology import *
from Options import * 
from Dataset import *
def main():

	"""
	Set up parameters
	"""
	args = parser.parse_args()

	"""
	Acquire data
	"""
	print('======================= Acquiring data ==============================')
	Data = GetData(args, Base = 0)
	BaseData = GetData(args, Base = 1)
	
	"""
	Clustering data
	"""
	print('========================= Clustering data ==============================')
	Loc, CData = Clustering(Data, args)
	BaseLoc, BaseCData = Clustering(BaseData, args)

	"""
	Get ripsers
	"""
	print('========================= Computing rips ==============================')
	BaseRips = GetTopologicalRips(BaseCData, args, Base = 1)
	Rips = GetTopologicalRips(CData, args, Base = 0)
	
	"""
	Get Betti number 
	"""
	print('======================= Computing barcodes =========================')
	BaseBarcode = GetBetti(BaseRips, args, Base = 1)
	Barcode = GetBetti(Rips, args, Base = 0)

	"""
	Compute Bottleneck distance or wasserstein distance
	"""
	print('======================= Computing distance between persistence diagrams =========================')
	GetPDDist(BaseRips[0], Rips, args)


if __name__ == '__main__':
	main()