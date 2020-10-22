import numpy as np 
import sys
import os
import pdb
sys.path.append(os.getcwd() + "/TopologyComparison/")
from Topology import *
from Options import * 


def main():
	args = parser.parse_args()

	print('======================= Loading Ripsers ==============================')
	BaseCData = None
	CData = None
	Label = [0.1, 0.3, 0.5]
	BaseRips = []
	Rips = []
	for i in range(len(Label)):
		args.W = Label[i]; args.Comp = 'VR'; args.Graph = 'Radius'; args.Load = True
		BaseRips.append(GetTopologicalRips(BaseCData, args, Base = 1))
		Rips.append(GetTopologicalRips(CData, args, Base = 0))
	print('======================= Drawing bottleneck ==============================')
	 
	for baserip, rip in zip(BaseRips, Rips):
		Dist0List = []
		Dist1List = []
	
		dist0, dist1 = GetPDDist(baserip[0], rip, args)
		Dist0List.append(dist0)
		Dist1List.append(dist1)
	pdb.set_trace()
	GetPDDistTogether(args, Dist0List, Dist1List, Label)

if __name__ == '__main__':
	main()