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

def SetScatterProp(ax, xn=None, yn=None, title=None, legend = True):
	fontsize = 14
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(2.5)

	ax.set_yticklabels([])
	ax.set_xticklabels([])

	if legend:   
		ax.legend(loc='upper right', shadow=True, prop={'weight':'bold'})

	if title != None:
		ax.set_title(title, fontweight='bold')
	return ax

def DrawOverlap(Args, Data, R, Index, Base = 0):
	"""
	Args: parser(). Parameter options
	Data: list of array. Dataset
	Index: list of array. Index of overlap data
	Base: int. Base case or nonbase case
	R: float. Radius/NN number to construct a graph
	"""
	if Base == 0:
		print('Drawing overlap region for nonbase case...')
		Path = os.getcwd() + '/Figures/%s/%s/Tau%.2f/OverlapWidth%.2f/%s/Overlap/%s/'%(Args.Exp, Args.DataType, Args.Tau, Args.W, Args.Comp, Args.Graph)
	elif Base == 1:
		print('Drawing overlap region for base case...')
		Path = os.getcwd() + '/Figures/%s/%s/Tau%.2f/OverlapWidth%.2f(Base)/%s/Overlap/%s/'%(Args.Exp, Args.DataType, Args.Tau, Args.BW, Args.Comp, Args.Graph)
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
		Fig.savefig(Path + 'R%.2fOverlap.png'%R, bbox_inches='tight')
	elif Args.Graph == 'NN':
		Fig.savefig(Path + 'K%dOverlap.png'%R, bbox_inches='tight')
	plt.close('all')

def DrawBarcode(Args, Barcode, R, Base = 0):
	"""
	Args: parser(). parameter options
	Barcode: Dictionary. Barcode
	Base: int. Base case or nonbase case
	R: float. Radius/NN number to construct a graph
	"""
	if Base == 0:
		print('Drawing betti number for nonbase case...')
		Path = os.getcwd() + '/Figures/%s/%s/Tau%.2f/OverlapWidth%.2f/%s/Bettinumber/%s/NT%.2f/'%(Args.Exp, Args.DataType, Args.Tau, Args.W, Args.Comp, Args.Graph, Args.NT)
		if not os.path.exists(Path):
			os.makedirs(Path)
	elif Base == 1:
		print('Drawing betti number for base case...')
		Path = os.getcwd() + '/Figures/%s/%s/Tau%.2f/OverlapWidth%.2f(Base)/%s/Bettinumber/%s/NT%.2f/'%(Args.Exp, Args.DataType, Args.Tau, Args.BW, Args.Comp, Args.Graph, Args.NT)
		if not os.path.exists(Path):
			os.makedirs(Path)
	Scale = np.linspace(Args.S, Args.L, Args.N)

	Fig1 = plt.figure(); Fig2 = plt.figure(); 
	ax1 = Fig1.gca(); ax2 = Fig2.gca(); 
	ax1.plot(Scale, Barcode[0], label = r'$\beta_0$', linewidth = 3); ax2.plot(Scale, Barcode[1], label = r'$\beta_1$', linewidth = 3)
	ax1 = SetPltProp(ax1, 'Scale',legend = True); ax2 = SetPltProp(ax2, 'Scale',legend = True) 

	if Args.Graph == 'Radius':
		Fig1.savefig(Path + 'R%.2fBetti0.png'%R, bbox_inches='tight')
		Fig2.savefig(Path + 'R%.2fBetti1.png'%R, bbox_inches='tight')
	elif Args.Graph == 'NN':
		Fig1.savefig(Path + 'K%dBetti0.png'%R, bbox_inches='tight')
		Fig2.savefig(Path + 'K%dBetti1.png'%R, bbox_inches='tight')
	plt.close('all')

def DrawPD(Args, rips, R, base = 0):
	"""
	Args: parser(). parameter options
	ripers: List. a list of ripser
	base: int. Base case or nonbase case
	R: float. Radius/NN number to construct a graph
	"""

	if base == 1:
		print('Drawing persistence diagram for base case...')
		Path = os.getcwd() + '/Figures/%s/%s/Tau%.2f/OverlapWidth%.2f(Base)/%s/PD/%s/NT%.2f/'%(Args.Exp, Args.DataType, Args.Tau, Args.BW, Args.Comp, Args.Graph, Args.NT)

	elif base == 0:
		print('Drawing persistence diagram for nonbase case...')
		Path = os.getcwd() + '/Figures/%s/%s/Tau%.2f/OverlapWidth%.2f/%s/PD/%s/NT%.2f/'%(Args.Exp, Args.DataType, Args.Tau, Args.W, Args.Comp, Args.Graph, Args.NT)

	if not os.path.exists(Path):
		os.makedirs(Path)
	if os.path.isfile(Path + ''):
	        data = np.load(Path)
	for i, d in enumerate(rips):
		# Figure1 = plot_diagrams(d['dgms'], plot_only=[0])
		# Figure2 = plot_diagrams(d['dgms'], plot_only=[1])
		# ax1 = Figure1.gca()
		# ax2 = Figure2.gca()
		# ax1 = SetPltProp(ax1)
		# ax2 = SetPltProp(ax2)
		# Figure1.savefig(Path + 'K%dCluster%dPD0.png'%(Args.K, i), bbox_inches='tight')
		# Figure2.savefig(Path + 'K%dCluster%dPD1.png'%(Args.K, i), bbox_inches='tight')
		Figure1 = plot_diagrams(d['dgms'])	
		ax1 = Figure1.gca()	
		ax1 = SetPltProp(ax1, grid = False, legend = True, pos = 'lower right')
		if Args.Graph == 'Radius':
			Figure1.savefig(Path + 'R%.2fCluster%dPD.png'%(R, i), bbox_inches='tight')
		elif Args.Graph == 'NN':
			Figure1.savefig(Path + 'K%dCluster%dPD.png'%(R, i), bbox_inches='tight')
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

def DrawPDDist(Args, Dist0, Dist1, c):
	"""
	Args: parser(). Parameter options
	Dist0: array. Distance bewteen base H0 diagram and the other H0 diagram 
	Dist1: array. Distance bewteen base H1 diagram and the other H1 diagram
	c: cluster number
	"""
	print('Drawing persistence diagram...')
	Path = os.getcwd() + '/Figures/%s/%s/Tau%.2f/OverlapWidth%.2f/%s/%s/%s/'%(Args.Exp, Args.DataType, Args.Tau, Args.W, Args.Comp, Args.Dist, Args.Graph)
	if not os.path.exists(Path):
		os.makedirs(Path)
	Fig1 = plt.figure(); ax1 = Fig1.gca()
	Fig2 = plt.figure(); ax2 = Fig2.gca()
	if Args.Graph == 'Radius':
		S = Args.R
	elif Args.Graph == 'NN':
		S = Args.K 

	ax1.plot(S, Dist0, label = 'H0', marker = 'o', markersize = 8, linewidth = 3); ax2.plot(S, Dist1, marker = 'o', markersize = 8,label = 'H1', linewidth = 3)
	if Args.Graph == 'Radius':
		ax1 = SetPltProp(ax1, 'R',legend = True); ax2 = SetPltProp(ax2, 'R',legend = True)
	elif Args.Graph == 'NN':
		ax1 = SetPltProp(ax1, 'K',legend = True); ax2 = SetPltProp(ax2, 'K',legend = True)
	Fig1.savefig(Path + 'DistCluster%dH0.png'%c, bbox_inches='tight')
	Fig2.savefig(Path + 'DistCluster%dH1.png'%c, bbox_inches='tight')
	plt.close('all')

def GetPDDistTogether(Args, DistList0, DistList1, Label):
	"""
	Args: parser(). Parameter options
	DistList0: list. A list of H0 distance
	DistList1: list. A list of H1 distance
	Label: list. Labels of each curve.
	"""
	print('Drawing persistence diagram for base case...')
	Path = os.getcwd() + '/Figures/%s/%s/Tau%.2f/OverlapWidth%.2f(Base)/%s/Dist/%s/'%(Args.Exp, Args.DataType, Args.Tau, Args.W, Args.Comp, Args.Graph)
	
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