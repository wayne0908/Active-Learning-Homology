import numpy as np 
import random 
import matplotlib.pyplot as plt
import scipy.io as sio 
import multiprocessing as mp
import pickle
import pdb
import os
# import sys
# sys.path.append(os.getcwd() + "/LabelQuery/")
from scipy.sparse import csr_matrix 
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import minimum_spanning_tree
import matlab.engine

def PassiveQuery(args, DataList, StatsPath):
	"""
	DataList: a list of clustered data
	"""
	Path = StatsPath + 'QueryIndex/'
	if not os.path.exists(Path):
		os.makedirs(Path)
	print('Query by passive scheme...')
	QueryIndexList = []
	for d in DataList:
		QueryIndexList.append(random.sample(list(np.arange(0, len(d))), len(d)))
	with open(Path + 'QueryIndex%d.txt'%(args.Trial), 'wb') as rp:
		pickle.dump(QueryIndexList, rp)
	return QueryIndexList

def S2Query(Args, DataList, StatsPath):
	"""
	DataList: a list of clustered data
	"""
	print('Query by active scheme...')
	Path = StatsPath + 'QueryIndex/'
	if not os.path.exists(Path):
		os.makedirs(Path)
	if Args.LoadQueryIndex == 1 and os.path.isfile(Path + 'QueryIndex%d.txt'%(Args.Trial)):
		print('Load successfully!')
		with open(Path + 'QueryIndex%d.txt'%(Args.Trial), 'rb') as fp:
			QueryIndexList = pickle.load(fp)
	else:
		eng = matlab.engine.start_matlab()
		eng.addpath(os.getcwd() + "/LabelQuery/", nargout=0)
		eng.edit('S2LabelQuery',nargout=1)
		GraphList = []; QueryIndexList = []
		for data in DataList:
			Graph = DistanceMetric(data, Args.r)
			GraphList.append(Graph)
			MatGraph = matlab.double(Graph.tolist()); MatData = matlab.double(data.tolist()) 
			QueryIndex = np.array(eng.S2LabelQuery(MatGraph, MatData)[0])-1 # index in matlab starting from one therefore minus one here.
			QueryIndexList.append(QueryIndex)
		with open(Path + 'QueryIndex%d.txt'%(Args.Trial), 'wb') as rp:
			pickle.dump(QueryIndexList, rp)
	return QueryIndexList
def DistanceMetric(Data, d):
	"""
	Data: array. Distance matrix
	d: float. Edge threshold.
	"""
	Feat = Data[:, :-1]
	Dist = pairwise_distances(Feat, metric="euclidean",n_jobs=mp.cpu_count())
	G = Dist <=d * np.mean(Dist)
	return G

