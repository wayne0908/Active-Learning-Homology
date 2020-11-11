import numpy as np 
import os
import pdb
import matplotlib.pyplot  as plt 
from persim import plot_diagrams
from sklearn.cluster import KMeans

def Clustering(Data, Args):
	"""
	Data: array. Dataset
	C: int. Cluster number
	"""
	print('Clustering to %d clusters...'%Args.C)
	CData = []
	Feat = Data[:, :-1]
	kmeans = KMeans(n_clusters = Args.C, random_state = 0).fit(Feat)
	Loc = kmeans.labels_
	for i in range(Args.C):
		CData.append(Data[Loc == i])
	return Loc, CData


def PartitionQuery(Data, Index, Loc, Args):
	"""
	Data: array. Dataset.
	Index: array. Query index.
	Loc: array. Cluster index
	Arg: parser(). Option parameters.
	"""
	print('Query type: %s. Partitioning queries...'%Args.Q)
	if max(Index) == len(Data):
		Index-=1

	PData = []
	Interval = int(len(Data) / Args.P)
	for i in range(Args.P):
		if i == Args.N -1:
			PData.append(Data)
		else:
			PIndex = np.sort(Index[: (i + 1) * Interval])
		PLoc = Loc[PIndex]
		CData = []
	
		for c in np.unique(PLoc):
			CData.append(Data[PIndex[PLoc == c]])
		PData.append(CData)
	return PData


def SetPltProp(ax, xn = None, title=None, grid = True, legend = True, pos = 'upper left'):
	fontsize = 14
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(2.5)

	for tick in ax.xaxis.get_major_ticks():
	    tick.label1.set_fontsize(fontsize)
	    tick.label1.set_fontweight('bold')
	for tick in ax.yaxis.get_major_ticks():
	    tick.label1.set_fontsize(fontsize)
	    tick.label1.set_fontweight('bold') 
	if legend:   
		ax.legend(loc=pos, shadow=True, prop={'weight':'bold'})
	if grid:
		ax.grid(linewidth='1.5', linestyle='dashed')
	if xn != None:
		ax.set_xlabel(xn, fontweight='bold')
	if title != None:
		ax.set_title(title, fontweight='bold')
	return ax

def SetScatterProp(ax, xn=None, yn=None, title=None, legend = True, Loc = 'upper right'):
	fontsize = 14
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(2.5)

	ax.set_yticklabels([])
	ax.set_xticklabels([])

	if legend:   
		ax.legend(loc=Loc, shadow=True, prop={'weight':'bold'})

	if title != None:
		ax.set_title(title, fontweight='bold')
	return ax

def DrawOverlap(Args, Data, P, R, Index, FigurePath):
	"""
	Args: parser(). Parameter options
	Data: list of array. Dataset
	Index: list of array. Index of overlap data
	R: float. Radius/NN number to construct a graph
	P: float. Proportion of used unlabelled data pool
	FigurePath: str. Path to figure.
	"""
	print('Drawing overlap region...')
	Path = FigurePath + 'Overlap/'
	if not os.path.exists(Path):
		os.makedirs(Path)
	Fig = plt.figure(); ax = Fig.gca()
	for CData, CIndex in zip(Data, Index):
		Feat = CData[:, :-1]
		Label = CData[:, -1].copy()
		Label[CIndex] = 2
		ax.scatter(Feat[Label == 0, 0], Feat[Label == 0, 1], c = 'r', label = 'Class 0', s = 30)
		ax.scatter(Feat[Label == 1, 0], Feat[Label == 1, 1], c = 'b', label = 'Class 1', s = 30)
		ax.scatter(Feat[Label == 2, 0], Feat[Label == 2, 1], c = 'g', label = 'Cut-set boundary', s = 30)
	ax = SetScatterProp(ax, legend = True)
	plt.axis('equal')
	if Args.Graph == 'Radius':
		Fig.savefig(Path + 'R%.2fPer%.2fOverlap.png'%(R, P), bbox_inches='tight')
	elif Args.Graph == 'NN':
		Fig.savefig(Path + 'K%dPer%.2fOverlap.png'%(R, P), bbox_inches='tight')
	plt.close('all')

def DrawBarcode(Args, Barcode, P, FigurePath):
	"""
	Args: parser(). parameter options
	Barcode: Dictionary. Barcode
	Base: int. Base case or nonbase case
	P: float. Proportion of unlabelled data used
	FigurePath: str. Path to figure.
	"""

	print('Drawing betti number...')
	
	Path = FigurePath + 'NT%.2f/Bettinumber/'%Args.NT
	if not os.path.exists(Path):
		os.makedirs(Path)
	Scale = np.linspace(Args.S, Args.L, Args.N)

	Fig1 = plt.figure(); Fig2 = plt.figure(); 
	ax1 = Fig1.gca(); ax2 = Fig2.gca(); 
	ax1.plot(Scale, Barcode[0], label = r'$\beta_0$', linewidth = 3); ax2.plot(Scale, Barcode[1], label = r'$\beta_1$', linewidth = 3)
	ax1 = SetPltProp(ax1, 'Scale',legend = True); ax2 = SetPltProp(ax2, 'Scale',legend = True) 

	if Args.Graph == 'Radius':
		Fig1.savefig(Path + 'R%.2fPer%.2fBetti0.png'%(Args.R, P), bbox_inches='tight')
		Fig2.savefig(Path + 'R%.2fPer%.2fBetti1.png'%(Args.R, P), bbox_inches='tight')
	elif Args.Graph == 'NN':
		Fig1.savefig(Path + 'K%dPer%.2fBetti0.png'%(Args.K, P), bbox_inches='tight')
		Fig2.savefig(Path + 'K%dPer%.2fBetti1.png'%(Args.K, P), bbox_inches='tight')
	plt.close('all')

def DrawPD(Args, rips, P, R, FigurePath):
	"""
	Args: parser(). parameter options
	ripers: List. a list of ripser
	base: int. Base case or nonbase case
	P: float. Proportion of used unlabelled data
	FigurePath: str. Path to figure
	"""

	print('Drawing persistence diagram...')

	Path = FigurePath + 'PD/'
	if not os.path.exists(Path):
		os.makedirs(Path)
	Dgms = [np.zeros((0, 2)), np.zeros((0, 2))]
	for i, d in enumerate(rips):		
		for j, dg in enumerate(d['dgms']):
			Dgms[j] = np.vstack([Dgms[j], dg])

	Figure1 = plot_diagrams(Dgms)	
	ax1 = Figure1.gca()	
	ax1 = SetPltProp(ax1, grid = False, legend = True, pos = 'lower right')
	if Args.Graph == 'Radius':
		# Figure1.savefig(Path + 'R%.2fPer%.2fCluster%dPD.png'%(R, P, i), bbox_inches='tight')
		Figure1.savefig(Path + 'R%.2fPer%.2fPD.png'%(R, P), bbox_inches='tight')
	elif Args.Graph == 'NN':
		# Figure1.savefig(Path + 'K%dPer%.2fCluster%dPD.png'%(R, P, i), bbox_inches='tight')
		Figure1.savefig(Path + 'K%dPer%.2fPD.png'%(R, P), bbox_inches='tight')
	plt.close('all')

def DrawData(Args, data, Base = 0):
	"""
	Args: parser(). Parameter options
	data: array. Dataset
	Base: int. Base case or nonbase case
	"""
	print('Drawing synthetic dataset...')
	Path = os.getcwd() + '/Figures/Dataset/%s/'%Args.DataType
	if not os.path.exists(Path):
		os.makedirs(Path)
	Fig = plt.figure(); ax = Fig.gca()

	Feat = data[:, :-1]
	Label = data[:, -1]
		
	ax.scatter(Feat[Label == 0, 0], Feat[Label == 0, 1], c = 'r', label = 'Class 0', s = 30)
	ax.scatter(Feat[Label == 1, 0], Feat[Label == 1, 1], c = 'b', label = 'Class 1', s = 30)
	plt.axis('equal')
	ax = SetScatterProp(ax, legend = True)
	if Base == 0:
		Fig.savefig(Path + 'Tau%.2fOvelap%.2f.png'%(Args.Tau, Args.W), bbox_inches='tight')
	elif Base == 1:
		Fig.savefig(Path + 'Tau%.2fOvelap%.2f.png'%(Args.Tau, Args.BW), bbox_inches='tight')
	plt.close('all')

def DrawPDDist(Args, Dist0, Dist1, FigurePath):
	"""
	Args: parser(). Parameter options
	Dist0: array. Distance bewteen base H0 diagram and the other H0 diagram 
	Dist1: array. Distance bewteen base H1 diagram and the other H1 diagram
	FigurePath: str. Path to figure
	"""
	print('Drawing persistence diagram...')
	Per = np.arange(0.05, Args.Per + 0.05, 0.05)
	Path =FigurePath + Args.Dist
	if not os.path.exists(Path):
		os.makedirs(Path)
	Fig1 = plt.figure(); ax1 = Fig1.gca()
	Fig2 = plt.figure(); ax2 = Fig2.gca()
	if Args.Graph == 'Radius':
		S = Args.R
	elif Args.Graph == 'NN':
		S = Args.K 

	ax1.plot(Per, Dist0, label = 'H0', marker = 'o', markersize = 8, linewidth = 3); ax2.plot(Per, Dist1, marker = 'o', markersize = 8,label = 'H1', linewidth = 3)

	ax1 = SetPltProp(ax1, 'Proportion',legend = True, pos = 'upper right'); ax2 = SetPltProp(ax2, 'Proportion',legend = True, pos = 'upper right')

	Fig1.savefig(Path + '/DistH0.png', bbox_inches='tight')
	Fig2.savefig(Path + '/DistH1.png', bbox_inches='tight')
	plt.close('all')

def DrawQueriedLabels(Args, QueryIndex, DataList, FigurePath):
	"""
	QueryIndex: a list query index for each cluster
	DataList: a list of clustered data
	FigurePath: str. Path to figure
	"""
	print('Drawing labelled data...')
	Path = FigurePath + 'QueriedLabel/'

	if not os.path.exists(Path):
		os.makedirs(Path)
	Per = np.arange(0.05, Args.Per + 0.05, 0.05)
	# DataListCopy = list(DataList)
	for p in Per:
		Fig = plt.figure()
		ax = plt.gca()
		for i in range(len(DataList)):
			d = DataList[i]; q = QueryIndex[i]
			if i == 0:
				ax.scatter(d[d[:, -1] == 0, 0], d[d[:, -1] == 0, 1], c= 'r', label = 'Class 0', s = 30)
				ax.scatter(d[d[:, -1] == 1, 0], d[d[:, -1] == 1, 1], c = 'b', label = 'Class 1', s = 30)
				ax.scatter(d[np.uint16(q[:int(len(q) * p)]), 0], d[np.uint16(q[:int(len(q) * p)]), 1], c = 'g', label = 'Queried data', s = 30)
			else:
				ax.scatter(d[d[:, -1] == 0, 0], d[d[:, -1] == 0, 1], c= 'r')
				ax.scatter(d[d[:, -1] == 1, 0], d[d[:, -1] == 1, 1], c = 'b')
				ax.scatter(d[np.uint16(q[:int(len(q) * p)]), 0], d[np.uint16(q[:int(len(q) * p)]), 1], c = 'g')
		ax = SetScatterProp(ax, Loc = 'upper left')
		Fig.savefig(Path + '/Per%.2fQueriedLabel.png'%p, bbox_inches='tight')
		plt.close('all')		
def GetPDDistTogether(Args, DistList0, DistList1, Label):
	"""
	Args: parser(). Parameter options
	DistList0: list. A list of H0 distance
	DistList1: list. A list of H1 distance
	Label: list. Labels of each curve.
	"""
	print('Drawing persistence diagram for base case...')
	Path = os.getcwd() + '/Figures/%s/%s/%s/Dist/%s/'%(Args.DataType, Args.Comp, Args.Graph)
	
	if not os.path.exists(Path):
		os.makedirs(Path)
	Fig1 = plt.figure(); ax1 = Fig1.gca()
	Fig2 = plt.figure(); ax2 = Fig2.gca()
	if Args.Graph == 'Radius':
		S = Args.R
	elif Args.Graph == 'NN':
		S = Args.K 

	count = 0
	for Dist0, Dist1 in zip(DistList0, DistList1):
		ax1.plot(S, Dist0, label = str(Label[count]), marker = 'o', markersize = 8, linewidth = 3); ax2.plot(S, Dist1, marker = 'o', markersize = 8,label = str(Label[count]), linewidth = 3)
		count+=1
	if Args.Graph == 'Radius':
		ax1 = SetPltProp(ax1, 'R',legend = True); ax2 = SetPltProp(ax2, 'R',legend = True)
	elif Args.Graph == 'NN':
		ax1 = SetPltProp(ax1, 'K',legend = True); ax2 = SetPltProp(ax2, 'K',legend = True)
	Fig1.savefig(Path + 'DistCluster0H0.png', bbox_inches='tight')
	Fig2.savefig(Path + 'DistCluster0H1.png', bbox_inches='tight')
	plt.close('all')

def GetData(Args):
	if Args.DataType == 'Syn':
		Dir = os.getcwd() + '/Data/Synthetic/MultiCompData5.npy'
	elif Args.DataType == 'Real':
		Dir = os.getcwd() + '/Data/Synthetic/MultiCompData5.npy'
	Data = np.load(Dir)
	return Data 

def GetDirectory(Args, Name):
	"""
	Name: str. Directory name, e.g.Passive or Active. 
	"""
	print('Getting %s directory...'%Name)
	if Args.Graph == 'Radius':
		if Name == 'Active':
			StatsPath = os.getcwd() + '/Stats/%s/%s/Radius(S2)=%.2f/%s/%s/Radius=%.2f/'%(Args.DataType, Name, Args.r, Args.Comp, Args.Graph, Args.R)
			FigurePath = os.getcwd() + '/Figure/%s/%s/Radius(S2)=%.2f/%s/%s/Radius=%.2f/'%(Args.DataType, Name, Args.r, Args.Comp, Args.Graph, Args.R)
		elif Name == 'Passive':
			StatsPath = os.getcwd() + '/Stats/%s/%s/%s/%s/Radius=%.2f/'%(Args.DataType, Name, Args.Comp, Args.Graph, Args.R)
			FigurePath = os.getcwd() + '/Figure/%s/%s/%s/%s/Radius=%.2f/'%(Args.DataType, Name, Args.Comp, Args.Graph, Args.R)			
	elif Args.Graph =='NN':
		if Name == 'Active':
			StatsPath = os.getcwd() + '/Stats/%s/%s/Radius(S2)=%.2f/%s/%s/Neighbor numbers=%d/'%(Args.DataType, Name, Args.r, Args.Comp, Args.Graph, Args.K)
			FigurePath = os.getcwd() + '/Figure/%s/%s/Radius(S2)=%.2f/%s/%s/Neighbor numbers=%d/'%(Args.DataType, Name, Args.r, Args.Comp, Args.Graph, Args.K)
		elif Name == 'Passive':
			StatsPath = os.getcwd() + '/Stats/%s/%s/%s/%s/Neighbor numbers=%d/'%(Args.DataType, Name, Args.Comp, Args.Graph, Args.K)
			FigurePath = os.getcwd() + '/Figure/%s/%s/%s/%s/Neighbor numbers=%d/'%(Args.DataType, Name, Args.Comp, Args.Graph, Args.K)
	if Name == 'Passive':
		FigurePath2 = os.getcwd() + '/Figure/%s/%s/'%(Args.DataType, Name)
	elif Name == 'Active':
		FigurePath2 = os.getcwd() + '/Figure/%s/%s/Radius(S2)=%.2f/'%(Args.DataType, Name, Args.r)
	return StatsPath, FigurePath, FigurePath2

