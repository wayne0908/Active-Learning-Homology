import numpy as np 
import multiprocessing as mp
import pdb
import os
import sys
import persim
import pickle
from Utility import *
from ripser import ripser 
from persim import plot_diagrams
from scipy.sparse import csr_matrix, csgraph
from sklearn.metrics import pairwise_distances
from supVR import compute_C_first, update_P, init_C_inds, compute_C_closest

def getIntervals(birth_values, death_values, dims, dim):

    return np.column_stack([birth_values[dims == dim], 
                            death_values[dims == dim]])

def bettiCounts(dims, birth_values, death_values, filtValues, dim):


    bettiCounts = []

    N = len(filtValues)
    bettiCounts = np.zeros(N)
    intervals = getIntervals(birth_values, death_values, dims, dim)
 
    if intervals.size:
        for i in range(N):
            bettiCounts[i] = np.dot((intervals[:, 0] <= filtValues[i]).astype(int), \
                                    (intervals[:, 1] > filtValues[i]).astype(int))

    return bettiCounts

def multi_conn_comp(P, numFilt):
    """
    Estimate singly and multiply connected components from P

    ncc - total connected components
    singly_conn - singly connected components
    multi_conn - multiply connected components
    """

    multi_conn = np.zeros(numFilt)
    singly_conn = np.zeros(numFilt)
    ncc = np.zeros(numFilt)
    for i in range(1, numFilt+1):
        P2 = (P < i).astype(np.int64)
        P2s = csr_matrix(P2)
        ncc[i-1], ccl = csgraph.connected_components(P2s)
    
        #print(ncc, ccl)
        uv, cnts = np.unique(ccl, return_counts = True)
        singly_conn[i-1] = np.sum(cnts == 1)
        #rint(ncc, singly_conn)
        multi_conn[i-1] = ncc[i-1] - singly_conn[i-1]
    return ncc, singly_conn, multi_conn

def opposing_neighbors(x, y, N):

	""" 
	All nearest neighbors from opposing classes 
	(class other than that of the example considered)       
	N - maximum number of nearest neighbors from opposing class that
	    needs to be stored
	S - indices of neighbors (row-wise)
	DN - distances of neighbors (row-wise)
	x - feature
	y - label
	"""

	# D = distance.squareform(
	#             distance.pdist(x, metric = "euclidean"))

	D = pairwise_distances(x, metric="euclidean",n_jobs=mp.cpu_count())
	DN = -1.0*np.ones((x.shape[0], N))
	S = -1*np.ones((x.shape[0], N), dtype = np.int32)

	for i in range(x.shape[0]):
		yi = y[i]
		# Neighbor indices from other classes
		opp_ind = np.where(y != yi)[0]
		D_opp = D[i, opp_ind]

		n = np.minimum(len(D_opp), N)
		
		# Compute the distances and indices of nearest neighbors
		S[i, 0:n] = opp_ind[np.argsort(D_opp)[0:n]]
		DN[i, 0:n] = D[i, S[i,0:n]]
	
	return S, DN, D

def LSLabelDistanceMetric(x, y, k, rho):
		""" 
		Modified from Kalthi's code.
		nearest neighbors from opposing classes with local scale computed
		based on D and B
		(class other than that of the example considered)

		x - data points
		y - class membership
		k - nearest neighbors to be considered from opposing class for 
		    computing local scale
		rho - List of multipliers to be used when computing the graph with
		    local scale. For a value r from this, an edge will be created when
		    d(x_i, x_j) < r*local_scale_i*local_scale_j and i, j belong to
		    opposing classes
		N - max number of nearest neighbors from opposing classes (user parameter)

		2 hop neighbors are then computed to complete the triangles and form
		simplices

		SIMILAR to localscale_neighbors_opposing_r but uses Cython for speedup
		This involves restricting the maximum number of neighbors for each point to N
		"""
		LP = []
		numFilt = len(rho)
		nTriv = np.zeros(numFilt)
		m = x.shape[0]
		# Compute opposing neighbors
		N = 20
		S, DN, _ = opposing_neighbors(x, y, k) # S: index of N nearest oppostite neighbours. DN: distance to N opposite neighbors
		# pdb.set_trace()
		# Sigma array
		# if DN[:,k-1] < 0:
		# 	pdb.set_trace()
		sigarr = np.sqrt(DN[:,k-1].ravel())

		# Other inits
		C = -1*np.ones((m,k+1), dtype = np.int32) # C = -1*np.ones((m,N+1), dtype = np.int32)
		Pupd = np.zeros((m,m), dtype = np.int32)
		P = np.zeros((m,m), dtype = np.int32)
		inds = np.zeros(m, dtype = np.int32)

		#t1 = time()

		for j, r in enumerate(rho):
		    # Re-initialize some arrays
		  
		    if j > 0:
		        C.fill(-1)
		        inds.fill(0)
		        Pupd.fill(0)
		    
		    # Call Cython functions for updating P
		    init_C_inds(C, inds)
		    compute_C_closest(C, S, inds, DN, r, sigarr)
		    update_P(C, inds, Pupd, 1)
		    np.maximum(Pupd, Pupd.transpose(), out = Pupd)
		    np.add(P, Pupd, out = P)
		    LP.append(Pupd)
		    nTriv[j] = np.sum(inds == 1)
		    
		P = numFilt - P

		return P, LP  

def DistanceMetric(Data, Scale):
	"""
	Data: array. Distance matrix
	Scale: array. Filtering Scale
	"""
	Feat = Data[:, :-1]
	Dist = pairwise_distances(Feat, metric="euclidean",n_jobs=mp.cpu_count())
	S = 0 # distance sum
	C = 0 # number of point pairs
	# SDist = np.sort(Dist, axis = 1)
	# D = np.mean(SDist[:, 1]) # average distance to nearest neighbor
	NewDist = np.zeros(Dist.shape, dtype = np.uint)
	NumFilt = len(Scale)
	FineDist = []
	for i in range(NumFilt):
		TDist = Dist <=Scale[i]
		NewDist+=TDist
		FineDist.append(TDist)
	NewDist = NumFilt - NewDist
	return NewDist, FineDist

def RemoveBoundary(Data, G, k):
	"""
	Data: array. Distance matrix
	k: int. k opposite nearest-neighbors/radius. 
	G:str. Graph type
	"""

	Feat = Data[:, :-1]
	Lab = Data[:, -1]
	# S - k nearest opposite label index; DN - k nearst opposite label distance; Dist - distance matrix
	if G == 'Radius':
		Dist = pairwise_distances(Feat, metric="euclidean",n_jobs=mp.cpu_count())
		B = k
	elif G == 'NN':
		S, DN, Dist = opposing_neighbors(Feat, Lab, k) 
		B = np.min(DN[:, -1])
	OppLabs = Lab.reshape((-1, 1)) != np.repeat(Lab.reshape((1, -1)), len(Lab), axis = 0) # opposite labels are connected
	Dist2 = Dist<=B
	Index = np.sum(Dist2 * OppLabs, 1) > 0
	return Index

def LabelDistanceMetric(L, Dist, Scale):
	"""
	L: array. Label.
	Dist: array. Distance matrix
	Scale: array. Filtering scale
	"""
	print('Computing distance matrix.... ')
	NewDist = np.zeros(Dist.shape, dtype = np.uint)
	NumFilt = len(Scale)
	mask_mat = np.zeros((len(L), len(L)))
	
	for i in range(len(L)):
		mask_mat[i] = L[i] == L
	mask_mat = mask_mat.astype(bool)
	C1 = np.zeros(Dist.shape, dtype = np.uint)
	C1bool = np.zeros(Dist.shape, dtype = np.bool)
	
	pool = mp.Pool(mp.cpu_count())
	FineDist = [pool.apply(parallel, args=(s, Dist, mask_mat, C1, C1bool, NewDist)) for s in Scale]
	NewDist = np.sum(np.array(FineDist), 0)
	pool.close()

	NewDist = NumFilt - NewDist 
	return NewDist, FineDist

def GetLVRDiagram(Data, Scale, k):
	"""
	Data: array. Dataset.
	Scale: array. Filtration scale.
	k: int. opposite neighbours
	"""
	Feat = Data[:, :-1]
	Lab = Data[:, -1]
	k = np.min((np.sum(Lab == 0), np.sum(Lab == 1), k)) # opposite neighbour number
	if k > 0:
		NewDist, _ = LSLabelDistanceMetric(Feat, Lab, k, Scale)
		Diagrams = ripser(NewDist, distance_matrix=True, do_cocycles = True)
	else:
		Diagrams = None; NewDist=None
	return Diagrams, NewDist

def GetVRDiagram(Data, Scale):
	"""
	Data: array. Dataset.
	Scale: array. Filtration scale.
	K: int. Number of nearest neighbors
	"""
	Feat = Data[:, :-1]
	Lab = Data[:, -1]
	NewDist, FineDist = DistanceMetric(Data, Scale)
	Diagrams = ripser(NewDist, distance_matrix=True, do_cocycles = True)
	# pdb.set_trace()
	# Diagrams = ripser(Data[:, :-1], distance_matrix=False, do_cocycles = True)
	return Diagrams, FineDist

def GetTopologicalRips(Data, Args, Base=0):
	"""
	Data: A list of clustered data. Dataset
	Comp: str. Complex type
	Graph: str. Graph type
	Base: int. Base ripser options
	"""
	if Base == 0:
		Path = os.getcwd() + '/Figures/%s/%s/Tau%.2f/OverlapWidth%.2f/%s/PD/%s/'%(Args.Exp, Args.DataType, Args.Tau, Args.W, Args.Comp, Args.Graph)
		if Args.Graph == 'Radius':
			R = Args.R # radius
		elif Args.Graph =='NN':
			R = Args.K # opposite neighbour number
	elif Base == 1:
		Path = os.getcwd() + '/Figures/%s/%s/Tau%.2f/OverlapWidth%.2f(Base)/%s/PD/%s/'%(Args.Exp, Args.DataType, Args.Tau, Args.BW, Args.Comp, Args.Graph)
		if Args.Graph == 'Radius':
			R = [Args.BR] # radius
		elif Args.Graph =='NN':
			R = [Args.BK] # opposite neighbour number

	if not os.path.exists(Path):
		os.makedirs(Path)

	if os.path.isfile(Path + 'Ripsers.txt') and Args.LoadRipser:
		print('Load successfully!')
		with open(Path + 'Ripsers.txt', 'rb') as fp:
			LListRips = pickle.load(fp)
	else: 
		LListNewDist = []
		LListRips = []
		for r in R:
			if Args.Graph == 'NN':
				print('Base option:%d, Opposite label number: %d, complex type: %s, Getting persistence diagrams...'%(Base, r, Args.Comp))
			else:
				print('Base option:%d, Radius: %.2f, complex type: %s, Getting persistence diagrams...'%(Base, r, Args.Comp))
				# if Base == 0:
				# 	Args.S = (Args.BW / 2 + r + Args.Tau - np.sqrt((Args.W / 2 + r ) ** 2 + (Args.Tau)**2\
				# 	          - 6 * Args.Tau*(Args.W / 2 + r )))/2 + 1e-10 # calculated lowest epsilon
				# 	Args.L = ((np.sqrt(9) -np.sqrt(8)) + Args.Tau + np.sqrt((np.sqrt(9) -np.sqrt(8)) ** 2 + (Args.Tau)**2\
				# 	          - 6 * Args.Tau*(np.sqrt(9) -np.sqrt(8))))/2 - 1e-10# calculated largest epsilon
				# else:
				# 	Args.S = (Args.BW / 2 + r + Args.Tau - np.sqrt((Args.BW / 2 + r ) ** 2 + (Args.Tau)**2\
				# 	          - 6 * Args.Tau*(Args.BW / 2 + r )))/2 + 1e-10 # calculated lowest epsilon
				# 	Args.L = ((np.sqrt(9) -np.sqrt(8)) + Args.Tau + np.sqrt((np.sqrt(9) -np.sqrt(8)) ** 2 + (Args.Tau)**2\
				# 	          - 6 * Args.Tau*(np.sqrt(9) -np.sqrt(8))))/2 - 1e-10# calculated largest epsilon			
			Scale = np.linspace(Args.S, Args.L, Args.N)
			ListRips = []
			ListNewDist = []
			Index = []
			
			for CData in Data:
				Feat = CData[:, :-1]
				Lab = CData[:, -1]
				if Args.Comp == 'LVR':

					Rips, NewDist = GetLVRDiagram(CData, Scale, r)

				elif Args.Comp == 'VR':
					CIndex = RemoveBoundary(CData, Args.Graph, r)
					Rips, NewDist = GetVRDiagram(CData[CIndex], Scale)		
					Index.append(CIndex)
				ListRips.append(Rips)
				ListNewDist.append(NewDist)
			if Args.Comp == 'VR' and Args.DrawBoundary:
				DrawOverlap(Args, Data, r, Index, Base) # draw covering region
			if Args.DrawPD:
				DrawPD(Args, ListRips, r, Base) # draw persistence diagrams
			LListRips.append(ListRips)
			LListNewDist.append(ListNewDist)
		with open(Path + 'Ripsers.txt', 'wb') as rp:
			pickle.dump(LListRips, rp)
	return LListRips

def GetBetti(Rips, Args, Base = 0):
	"""
	Rips: List. List of ripsers
	Comp: str. Complex type
	NT: float. noise removel threshold.
	Base: int. Base or non base case
	"""
	print('Base:%d, Getting betti number...'%Base)
	if Base == 0:
		if Args.Graph == 'Radius':
			R = Args.R
		elif Args.Graph == 'NN':
			R = Args.K
	elif Base ==1:
		if Args.Graph == 'Radius':
			R = [Args.BR]
		elif Args.Graph == 'NN':
			R = [Args.BK]
	BettinumDic = {} # store betti number information where each key represents radius 
					 # and the content represent betti number in each number. 

	for ri, r in zip(Rips, R):
		BettinumDic[r] = []

		for ri1 in ri:

			Scale = np.linspace(Args.S, Args.L, Args.N)
			Bettinum = np.zeros((2, len(Scale)))
			
			if ri1 != None:
				if Args.Comp == 'LVR':
					CBettinum = LVRCountBetti2(ri1, Scale, Args.NT)
				elif Args.Comp == 'VR':	
					CBettinum = VRCountBetti(ri1, Scale, Args.NT)
				Bettinum[0] +=CBettinum[0]
				Bettinum[1] +=CBettinum[1]
			BettinumDic[r].append(Bettinum)
			DrawBarcode(Args, Bettinum, r)
	

	if Base == 0:
		Path = os.getcwd() + '/Figures/%s/%s/Tau%.2f/OverlapWidth%.2f/%s/Bettinumber/%s/NT%.2f/Dic/'%(Args.Exp, Args.DataType, Args.Tau, Args.W, Args.Comp, Args.Graph, Args.NT)
	elif Base == 1:
		Path = os.getcwd() + '/Figures/%s/%s/Tau%.2f/OverlapWidth%.2f(Base)/%s/Bettinumber/%s/NT%.2f/Dic/'%(Args.Exp, Args.DataType, Args.Tau, Args.BW, Args.Comp, Args.Graph, Args.NT)
	if not os.path.exists(Path):
		os.makedirs(Path)
	with open(Path + 'BettiTrial%d.txt'%(Args.Trial), 'wb') as rp:
		pickle.dump(BettinumDic, rp)
	return Bettinum



def LVRCountBetti1(rips, P, Scale, NT):
	print('Counting betti number...')
	"""
	This function remove trival homology groups in the zero-th dimension
	rips: dictionary. Contains the information about the digrams acquired from ripser function.
	P: array. Distance matrix.
	NT: float. noise removel threshold.
	Scale: array. Filtering scale
	"""
	NumFIl = len(Scale)
	dgm_dims = [dgm.shape[0] for dgm in rips['dgms']]
	dims = np.hstack([idx*np.ones(i) for idx, i in enumerate(dgm_dims)])
	dgms = np.vstack(rips['dgms'])
	birth_values = dgms[:,0].ravel()
	death_values = dgms[:,1].ravel()

	kid = (death_values - birth_values) > (NumFIl * NT) # keep useful homology features
	birth_values = birth_values[kid]
	death_values= death_values[kid]
	dims = dims[kid]


	ncc, nTriv, _ = multi_conn_comp(P, NumFIl) 
	filtValues = np.linspace(0, NumFIl, NumFIl)
	bc={};bc[0] = np.zeros(len(Scale));bc[1] = np.zeros(len(Scale))
	for dim in np.unique(dims):
		# print("dimension %d" % (dim))

		if dim == 0:
		    bc[dim] = ncc
		else:
		    bc[dim] = bettiCounts(dims, birth_values, death_values, filtValues, dim)
		if dim == 0:
			bc[dim] = np.array(bc[dim] - nTriv)
		bc[dim] = bettiCounts(dims, birth_values, death_values, filtValues, dim)
	return bc 

def LVRCountBetti2(rips, Scale, NT):
	print('Counting betti number...')
	"""
	rips: dictionary. Contains the information about the digrams acquired from ripser function.
	Scale: array. Filtering scale
	NT: float. noise removel threshold.
	"""
	NumFIl = len(Scale)
	dgm_dims = [dgm.shape[0] for dgm in rips['dgms']]
	dims = np.hstack([idx*np.ones(i) for idx, i in enumerate(dgm_dims)])
	dgms = np.vstack(rips['dgms'])
	birth_values = dgms[:,0].ravel()
	death_values = dgms[:,1].ravel()

	kid = (death_values - birth_values) > (NumFIl * NT) # keep useful homology features
	birth_values = birth_values[kid]
	death_values= death_values[kid]
	dims = dims[kid]

	filtValues = np.linspace(0, NumFIl, NumFIl)
	bc={};bc[0] = np.zeros(len(Scale));bc[1] = np.zeros(len(Scale))
	for dim in np.unique(dims):
		bc[dim] = bettiCounts(dims, birth_values, death_values, filtValues, dim)
	return bc 


def VRCountBetti(rips, Scale, NT):
	print('Counting betti number...')
	"""
	rips: dictionary. Contains the information about the digrams acquired from ripser function.
	Scale: array. Filtering scale
	NT: float. noise removel threshold.
	"""
	NumFIl = len(Scale)

	dgm_dims = [dgm.shape[0] for dgm in rips['dgms']]
	dims = np.hstack([idx*np.ones(i) for idx, i in enumerate(dgm_dims)])
	dgms = np.vstack(rips['dgms'])
	birth_values = dgms[:,0].ravel()
	death_values = dgms[:,1].ravel()

	kid = (death_values - birth_values) > (NumFIl  * NT) # keep useful homology features
	birth_values = birth_values[kid]
	death_values= death_values[kid]
	dims = dims[kid]

	filtValues = np.linspace(0, NumFIl, NumFIl)
	bc={};bc[0] = np.zeros(len(Scale));bc[1] = np.zeros(len(Scale))
	for dim in np.unique(dims):
		bc[dim] = bettiCounts(dims, birth_values, death_values, filtValues, dim)
	return bc

def GetPDDist(BaseRips, Rips, args):
	"""
	BaseRips: Dic. Base Ripser
	Rips: Dic. Different ripser information
	"""
	print('Getting %s distance...'%(args.Dist))

	if args.Graph == 'Radius':
		L = len(args.R)
	elif args.Graph == 'NN':
		L = len(args.K)


	Path = os.getcwd() + '/Figures/%s/%s/Tau%.2f/OverlapWidth%.2f/%s/%s/%s/Dic/'%(args.Exp, args.DataType, args.Tau, args.W, args.Comp, args.Dist, args.Graph)

	if not os.path.exists(Path):
		os.makedirs(Path)

	PDDist = {}
	for c in range(args.C):
		BaseDgms = BaseRips[c]['dgms']
		Dist0 = np.zeros(L)
		Dist1 = np.zeros(L)
		PDDist['Cluster%d'%c] = []
		for l in range(L):
			if args.Dist == 'Bottleneck':
				Dist0[l] = persim.bottleneck(BaseDgms[0], Rips[l][c]['dgms'][0])
				Dist1[l] = persim.bottleneck(BaseDgms[1], Rips[l][c]['dgms'][1])
			elif args.Dist =='Wasserstein':
				Dist0[l] =  persim.sliced_wasserstein(BaseDgms[0], Rips[l][c]['dgms'][0])
				Dist1[l] =  persim.sliced_wasserstein(BaseDgms[1], Rips[l][c]['dgms'][1])
			elif args.Dist == 'Heat':
				Dist0[l] =  persim.heat(BaseDgms[0], Rips[l][c]['dgms'][0])
				Dist1[l] =  persim.heat(BaseDgms[1], Rips[l][c]['dgms'][1])
		PDDist['Cluster%d'%c].extend((Dist0, Dist1))

		if args.DrawDist:
			DrawPDDist(args, Dist0, Dist1, c)
	with open(Path + 'DistTrial%d.txt'%(args.Trial), 'wb') as rp:
		pickle.dump(PDDist, rp)
	





	