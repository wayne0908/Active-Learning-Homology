import numpy as np
import os  
import pdb
import sys
import random
from Utility import *
# sys.path.append(os.getcwd())

class disk2D:
    """ Disk class for 2D """
    def __init__(self, c, r):
        """
            c - center (tuple)
            r - radius
        """
        self.c = c
        self.r = r
        
    def samples(self, ns):
        thetas = np.random.uniform(low = 0.0, high = 2.0*np.pi, size = ns)
        rs = np.random.uniform(low = 0.0, high = self.r, size = ns)
        samp = np.vstack((rs*np.cos(thetas), rs*np.sin(thetas))).T +np.array(self.c)
        return samp
    
class annulus2D:
    """ Annulus class for 2D """
    def __init__(self, c, rlow, rhigh):
        """
            c - center (tuple)
            rlow, rhigh - low and high radii
        """
        self.c = c
        self.rlow = rlow
        self.rhigh = rhigh
        
    def samples(self, ns):
        thetas = np.random.uniform(low = 0.0, high = 2.0*np.pi, size = ns)
        rs = np.random.uniform(low = self.rlow, high = self.rhigh, size = ns)
        samp = np.vstack((rs*np.cos(thetas), rs*np.sin(thetas))).T +np.array(self.c)
        return samp

class annuluswithhold2D:
    """ Annulus class for 2D """
    def __init__(self, c, rlow, rhigh, hcshift=0.6, hcratio = 0.4):
        """
            c - center (tuple)
            rlow, rhigh - low and high radii
            hcshift: hole center shift; hcratio: hole radius / (rhigh - rlow)
        """
        self.c = c
        self.rlow = rlow
        self.rhigh = rhigh
        self.hc = (c[0] + rlow  + (rhigh - rlow) * hcshift, c[1])
        self.hr = (rhigh - rlow) * hcratio
    
    def InHole(self, samp, c, r):
        In = False
        if (samp[0] - c[0]) ** 2 + (samp[1] - c[1]) ** 2 <= r**2:
            In = True
            
        return In 
    def samples(self, ns):
        i = 0
        samp = np.zeros((0, 2))
        while(i < ns):
            thetas = np.random.uniform(low = 0.0, high = 2.0*np.pi, size = 1)
            rs = np.random.uniform(low = self.rlow, high = self.rhigh, size = 1)
            onesamp = np.vstack((rs*np.cos(thetas), rs*np.sin(thetas))).T +np.array(self.c)
            
            if not self.InHole(onesamp[0], self.hc, self.hr):
                samp = np.vstack((samp, onesamp))
                i+=1
        return samp

class diskwithhold2D:
    """ Annulus class for 2D """
    def __init__(self, c, r, hcshift=0.6, hcratio = 0.2):
        """
            c - center (tuple)
            r - radius
            hcshift: hole center shift; hcratio: hole radius / (rhigh - rlow)
        """
        self.c = c
        self.r = r
        self.hc = (c[0] + r * hcshift, c[1])
        self.hr = r * hcratio
           
    def InHole(self, samp, c, r):
        In = False
        if (samp[0] - c[0]) ** 2 + (samp[1] - c[1]) ** 2 <= r**2:
            In = True            
        return In 
    def samples(self, ns):
        i = 0
        samp = np.zeros((0, 2))
        while(i < ns):
            thetas = np.random.uniform(low = 0.0, high = 2.0*np.pi, size = 1)
            rs = np.random.uniform(low = 0.0, high = self.r, size = 1)
            onesamp = np.vstack((rs*np.cos(thetas), rs*np.sin(thetas))).T +np.array(self.c)
            
            if not self.InHole(onesamp[0], self.hc, self.hr):
                samp = np.vstack((samp, onesamp))
                i+=1
        return samp

def CreateDataType1(Tau, W, N = 2000):
    """
    dataset of Disk + annulus
    Tau: float. Conditional number
    W: float. Area of overlap
    N: int. Sample number
    """
    print('Creating Type1 synthetic dataset with tau = %.2f and overlap width = %.2f...'%(Tau, W))
    c = (0, 0) # center of disk
    r = Tau # radius of disk
    
    rlow = Tau # inner radius of an annulus
    rhigh = 2 * Tau # exterior radius of an annulus

    OverlapRlow = Tau - W / 2 # inner radius of an annulus
    OverlapRHigh = Tau + W/2 # exterior radius of an annulus

    Disk = disk2D(c, r)
    Annulus = annulus2D(c, rlow, rhigh)
    OverlapArea = annulus2D(c, OverlapRlow, OverlapRHigh)

    SDisk = Disk.samples(N)
    SAnnulus = Annulus.samples(N)
    SOverlap = OverlapArea.samples(N)

    LabeledDisk = np.hstack((np.vstack((SDisk, SOverlap[: int(N/2)])), np.zeros(N + int(N/2)).reshape((-1, 1))))
    LabeledAnnulus = np.hstack((np.vstack((SAnnulus, SOverlap[int(N/2):])), np.ones(N + int(N/2)).reshape((-1, 1))))

    data = np.vstack((LabeledDisk, LabeledAnnulus))
    
    return data

def CreateDataType2(Tau, W, N = 1000):
    """
    dataset of Disk + annulus with a hole
    Tau: float. Conditional number
    W: float. Area of overlap
    N: int. Sample number
    """
    print('Creating Type2 synthetic dataset with tau = %.2f and overlap width = %.2f...'%(Tau, W))
    c = (0, 0) # center of disk
    r = Tau # radius of disk
    
    rlow = Tau # inner radius of an annulus
    rhigh = 1.5 * Tau # exterior radius of an annulus

    OverlapRlow = Tau - W / 2 # inner radius of an annulus
    OverlapRHigh = Tau + W/2 # exterior radius of an annulus

    Disk = diskwithhold2D(c, r)
    Annulus = annulus2D(c, rlow, rhigh)
    OverlapArea = annulus2D(c, OverlapRlow, OverlapRHigh)

    SDisk = Disk.samples(N)
    SAnnulus = Annulus.samples(N)
    SOverlap = OverlapArea.samples(N)

    LabeledDisk = np.hstack((np.vstack((SDisk, SOverlap[: int(N/2)])), np.zeros(N + int(N/2)).reshape((-1, 1))))
    LabeledAnnulus = np.hstack((np.vstack((SAnnulus, SOverlap[int(N/2):])), np.ones(N + int(N/2)).reshape((-1, 1))))

    data = np.vstack((LabeledDisk, LabeledAnnulus))

    # random.shuffle(data)
    return data

def GetData(Args, Base=0):
    """
    Args: parse(). option parameters
    Base:int. Base case or nonbase case
    """
    print('Base:%d, Acquire %s dataset (nonbase data)...'%(Base, Args.Exp))
    if Args.Exp == 'Real':
        data = np.load(Args.DataDir)
    elif Args.Exp == 'Synthetic':
        if Base == 0:
            W = Args.W
            Path = os.getcwd() + '/SyntheticDataset/%s/Tau%.2fOvelap%.2f.npy'%(Args.DataType, Args.Tau, W)
        elif Base == 1:
            W = Args.BW
            Path = os.getcwd() + '/SyntheticDataset/%s/Tau%.2fOvelap%.2f.npy'%(Args.DataType, Args.Tau, W)
        
        if os.path.isfile(Path) and Args.LoadData:
            data = np.load(Path)
        else:
            if Args.DataType == 'Type1':
                data = CreateDataType1(Args.Tau, W)
            elif Args.DataType == 'Type2':
                data = CreateDataType2(Args.Tau, W)
            if not os.path.exists(os.path.dirname(Path)):
                os.makedirs(os.path.dirname(Path))
            np.save(Path, data)
            DrawData(Args, data)
    return data 

