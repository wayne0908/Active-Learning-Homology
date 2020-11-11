import numpy as np 
import matplotlib.pyplot  as plt 
from matplotlib.ticker import FormatStrFormatter
import pdb

def GetIntersection(d, r1, r2):
	"""
	Get intersection of two circles
	d: distance of two circle centers
	r1: radius of the left circle
	r2: radius of the right circle 
	"""
	if r1 < r2:
		T = r1; r1 = r2; r2 = T
	d1 = (r1**2 - r2**2 + d**2)/(2*d); d2 = d - d1 
	A = r1 ** 2 * np.arccos(d1/r1) - d1 * np.sqrt(r1 ** 2 - d1 ** 2) + r2**2 * np.arccos(d2/r2) - d2 * np.sqrt(r2**2 - d2**2)
	return A 

def GetMeasure0(r, Tau, w, l1, l2):

	"""
	Measure for class 0
	r: radius of a covering ball
	Tau: 1/Tau is the condition number of the manifold. Also Tau can be 
	     regarded as the radius of a circle.
	w: the radius of a smallesGetMeasure0t neighborhood enclosing the overlap
	l1: width of the feature space/domain
	l2: length of the feature space/domain
	"""
	d = 1/ (np.pi * (Tau + w)**2)

	if r <= w:
		u = np.pi * r ** 2 * d 
	else:
		Intersect1 = GetIntersection(Tau, r, Tau + w)
		Intersect2 = GetIntersection(Tau, r, Tau - w)
		u = Intersect1 * d
	
	return u

def GetMeasure1(r, Tau, w, l1, l2):
	"""
	Measure for class 1
	r: radius of a covering ball
	Tau: 1/Tau is the condition number of the manifold. Also Tau can be 
	     regarded as the radius of a circle.
	w: the radius of a smallest neighborhood enclosing the overlap
	l1: width of the feature space/domain
	l2: length of the feature space/domain
	"""
	
	d = 1/(l1 * l2 - (np.pi * (Tau - w) ** 2))

	if r <= w:
		u = np.pi * r ** 2 * d 
	else:
		Intersect1 = GetIntersection(Tau, r, Tau + w)
		Intersect2 = GetIntersection(Tau, r, Tau - w)
		u = (np.pi * r **2 - Intersect2) * d
	return u

def GetMeasure(r, Tau, w, q, l1, l2):
	d1 = 1/(l1 * l2 - (np.pi * (Tau - w) ** 2)); d0 = 1/ (np.pi * (Tau + w)**2) 
	d = (1-q) * d1 + q * d0 

	if r<=w:
		u = np.pi * r**2 *d 
	else:
		Intersect1 = GetIntersection(Tau, r, Tau + w)
		Intersect2 = GetIntersection(Tau, r, Tau - w)
		u = (Intersect1 - Intersect2) * d  + (np.pi * r **2 - Intersect1) * d1 * (1-q) + Intersect2 * d0 * q
	return u 

def GetCoveringNum(r, R):
	"""
	Computing covering number 
	r: radius of a covering ball
	R: Radius of the circle to be covered
	"""
	N = np.pi/(np.arcsin(r/(R)))
	return N

def GetPassiveComplexity(Tau, w, q, l1 , l2, gamma, delta):
	"""
	Tau: 1/Tau is the condition number of the manifold. Also Tau can be 
	     regarded as the radius of a circle.
	q: Class weight for zero 
	w: the radius of a smallest neighborhood enclosing the overlap
	l1: width of the feature space/domain
	l2: length of the feature space/domain
	gamma: density parameter
	delta: failure probability
	"""
	rho0 = GetMeasure0(gamma/4, Tau, w, l1, l2)
	rho1 = GetMeasure1(gamma/4, Tau, w, l1, l2)
	N = GetCoveringNum(gamma/4, Tau)
	S1 = 1/((1-q) * rho0) * (np.log(2 * N) + np.log(1/(1 - np.sqrt(1-delta))))
	S0 = 1/(q * rho1) *(np.log(2 * N) + np.log(1/(1 - np.sqrt(1-delta))))
	# S1 = 1/(0.5 * rho0) * (np.log(2 * N) + np.log(1/(np.sqrt(1-delta))))
	# S0 = 1/(0.5 * rho1) *(np.log(2 * N) + np.log(1/(np.sqrt(1-delta))))
	return max(S1, S0)

def GetPassiveComplexity2(Tau, w, q, l1 , l2, gamma, delta):
	"""
	Tau: 1/Tau is the condition number of the manifold. Also Tau can be 
	     regarded as the radius of a circle. 
	q: Class weight for zero 
	w: the radius of a smallest neighborhood enclosing the overlap
	l1: width of the feature space/domain
	l2: length of the feature space/domain
	gamma: density parameter
	delta: failure probability
	"""

	print('=================== Simulating passive learning complexity ==============================')
	rho0 = GetMeasure0(gamma/4, Tau, w, l1, l2)
	rho1 = GetMeasure1(gamma/4, Tau, w, l1, l2)
	N = GetCoveringNum(gamma/4, Tau)
	S1 = 1/((1 - q) * rho0) * (np.log(2 * N) + np.log(1/delta))
	S0 = 1/(q * rho1) *(np.log(2 * N) + np.log(1/delta))
	print('Tau:%.4f, W:%.4f, gamma:%e, domain l1:%.2f, domain l2:%.2f,\
		   covering ball0 measure:%e, covering ball1 measure:%e, delta:%.2f, Sample complexity: %e'
		  %(Tau, w, gamma, l1, l2, rho0, rho1, delta, max(S1, S0)))
	# pdb.set_trace()
	return max(S1, S0)

def GetActiveQueryComplexity(Tau, w, q, l1 , l2, gamma, delta, beta = 0.5):
	"""
	Get query complexity 
	Tau: 1/Tau is the condition number of the manifold. Also Tau can be 
	     regarded as the radius of a circle. 
	q: Class weight for zero 
	w: the radius of a smallest neighborhood enclosing the overlap
	l1: width of the feature space/domain
	l2: length of the feature space/domain
	gamma: density parameter
	delta: failure probability
	beta: component weight
	"""

	print('=================== Simulating active learning query complexity ==============================')
	S = GetPassiveComplexity(Tau, w, q, l1 , l2, gamma, delta)
	h = GetMeasure(w + gamma, Tau, w, q, l1, l2)
	N = GetCoveringNum(gamma + w, Tau)
	D = np.log(1/(beta * (1-np.sqrt(1-delta))))/(np.log(1/(1-beta))) + S * N *h*(np.log2(S) + 1)

	print('Tau:%.4f, W:%.4f, gamma:%e, domain l1:%.2f, domain l2:%.2f,\
		   delta:%.2f, Covering number:%d, Covering ball measure: %e, Sample complexity: %e,\
		   Query complexity: %e, Q/S:%.2f'%
		   (Tau, w, gamma, l1, l2, delta, N, h, S, D, D/float(S)))
	
	return D

def SetPltProp(ax, xn = None, yn=None, title=None, grid = True, legend = True, pos = 'upper left'):
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
	if yn!= None:
		ax.set_ylabel(yn, fontweight='bold')
	if title != None:
		ax.set_title(title, fontweight='bold')
	return ax

def PlotComparison(T, S1, S2, xaxislabel):
	"""
	Comparison between sample complexity and query complexity under same delta (failure probability)
	T: a list of tau 
	S1: a list of sample complexity 
	S2: a list of query complexity
	"""
	Fig = plt.figure(); ax = Fig.gca(); 
	# ax.semilogy(T, S2, basey=2, c  = 'r', label = 'Active learning', marker = 'o', markersize = 8, linewidth=3)
	# ax.semilogy(T, S1, basey = 2,c  = 'b', label = 'Passive learning', marker = 'o', markersize = 8, linewidth=3)

	ax.plot(T, S2/S1,c  = 'b', label = '', marker = 'o', markersize = 8, linewidth=3)
	ax = SetPltProp(ax, xaxislabel, 'Query complexity to sample complexity', legend = False, pos = 'upper right')
	ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	Fig.show()
	pdb.set_trace()

def main():
	l1 = 5; l2 = 5; delta = 10**-8; beta = 10**-3; 
	FixTau = 0; # 1 for fixing tau otherwise fixing w 
	if FixTau == 1:
		Tau = 0.1
		W = np.linspace(10**-10, 1.6*10**-2, num=20)
		TAU = np.ones(len(W)) * Tau
	else:
		TAU = np.linspace(0.1,0.7,num=20);  
		W = np.ones(len(TAU)) * 10**-10;

	GAMMA = ((3-np.sqrt(8)) * TAU - W) * 0.99
	S1 = np.zeros(len(TAU)); S2 = np.zeros(len(TAU))
	for i in range(len(TAU)): 
		tau = TAU[i]; gamma = GAMMA[i]; w = W[i]; beta = np.pi * tau **2 / (l1 * l2) 
		S1[i] = GetPassiveComplexity2(tau, w, beta, l1 , l2, gamma, delta)
		S2[i] = GetActiveQueryComplexity(tau, w, beta, l1 , l2, gamma, delta, beta)
	# pdb.set_trace()
	if FixTau == 1:
		PlotComparison(W, S1, S2, 'w')
	else:
		PlotComparison(TAU, S1, S2, r'$\tau$')
if __name__ == '__main__':
	main()