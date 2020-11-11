import numpy as np 
import sys
import os
import pdb
sys.path.append(os.getcwd() + "/TopologyComparison/")
sys.path.append(os.getcwd() + "/LabelQuery/")
sys.path.append(os.getcwd() + "/CreatDataset/")
from Topology import *
from Options import * 
from LabelQuery import * 

def main():

	"""
	Set up parameters
	"""
	args = parser.parse_args()

	"""
	Acquire data
	"""
	print('======================= Acquiring data ==============================')
	Data = GetData(args)

	"""
	Clustering data
	"""
	print('========================= Clustering data ==============================')
	Loc, DataList = Clustering(Data, args)

	"""
	Bulid directory
	"""
	print('========================= Building directory ==============================')
	StatsPath1, FigurePath1, QueryFigurePath1 = GetDirectory(args, 'Passive')
	StatsPath2, FigurePath2, QueryFigurePath2 = GetDirectory(args, 'Active')

	"""
	S2 for label query  
	"""
	print('========================= Querying data ==============================')
	ActiveQueryIndex = S2Query(args, DataList, StatsPath2)
	PassiveQueryIndex = PassiveQuery(args, DataList, StatsPath1)

	"""
	Mark queried labels by images
	"""
	print('========================= Drawing queried examples ==============================')
	DrawQueriedLabels(args, PassiveQueryIndex, DataList, QueryFigurePath1)
	DrawQueriedLabels(args, ActiveQueryIndex, DataList, QueryFigurePath2)
	# pdb.set_trace()
	"""
	Get ripsers
	"""
	print('========================= Computing rips ==============================')
	PassiveRips = GetTopologicalRips(PassiveQueryIndex, DataList, args, StatsPath1, FigurePath1)
	ActiveRips = GetTopologicalRips(ActiveQueryIndex, DataList, args, StatsPath2, FigurePath2)
	"""
	Get Betti number 
	"""
	print('======================= Computing barcodes =========================')
	PassiveBarcode = GetBetti(PassiveRips, args, StatsPath1, FigurePath1)
	ActiveBarcode = GetBetti(ActiveRips, args, StatsPath2, FigurePath2)

	"""
	Compute Bottleneck distance or wasserstein distance
	"""
	print('======================= Computing distance between persistence diagrams =========================')
	GetPDDist(PassiveRips[-1], PassiveRips, args, StatsPath1, FigurePath1)
	GetPDDist(ActiveRips[-1], ActiveRips, args, StatsPath2, FigurePath2)

if __name__ == '__main__':
	main()