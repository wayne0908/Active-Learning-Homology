import argparse 
import numpy as np 

parser = argparse.ArgumentParser(description='Finding homology with active learning')

parser.add_argument('--N', type = int, default = 100, help = 'number of filtering')

parser.add_argument('--C', type = int, default = 1, help = 'Clustering number')

parser.add_argument('--Trial', type = int, default = 1, help = 'Trial number')

parser.add_argument('--S', type = float, default = 40, help = 'Smallest filtering scale')

parser.add_argument('--L', type = float, default = 40, help = 'Largest filtering scale')

parser.add_argument('--NT', type = int, default = 0, help = 'Noise removal threshold')

parser.add_argument('--K', type = list, default = [x for x in range(10, 251, 10)] , help = 'list of number of considered nearest opposite neighbors for nonbase case')

parser.add_argument('--R', type = list, default = [x for x in np.arange(0.1, 3, 0.05)], help = 'list of radius of graph for nonbase case')

parser.add_argument('--BK', type = int, default = 10, help = 'list of number of considered nearest opposite neighbors for nonbase case')

parser.add_argument('--LoadData', type = int, default = 0, help = 'load data')

parser.add_argument('--LoadRipser', type = int, default = 0, help = 'load ripsers indicator')

parser.add_argument('--DrawDist', type = int, default = 0, help = 'draw persistance diagrams distance or not')

parser.add_argument('--DrawPD', type = int, default = 0, help = 'draw persistance diagrams or not')

parser.add_argument('--DrawBoundary', type = int, default = 0, help = 'draw boundaries of cutset or not')

parser.add_argument('--BR', type = float, default = 0.1, help = 'list of radius of graph for nonbase case')

parser.add_argument('--Tau', type = float, default = 5, help = 'Conditional number')

parser.add_argument('--W', type = float, default = 0.3, help = 'Overlap width for nonbase data')

parser.add_argument('--BW', type = float, default = 0.05, help = 'Overlap width for base data')

parser.add_argument('--Graph', type = str, default = 'NN', help = 'Graph type')

parser.add_argument('--Dist', type = str, default = 'Bottleneck', help = 'ways to compare persistance diagrams')

parser.add_argument('--Exp', type = str, default = 'Synthetic', help = 'Synthetic or Real dataset experiments')

parser.add_argument('--DataType', type = str, default = 'Type2', help = 'Data type')

parser.add_argument('--Comp', type = str, default = 'VR', help = 'Check complex type')

parser.add_argument('--DataDir', type = str, default = '/home/weizhi/Desktop/Research/DecisionCapture/Dataset/Synthetic/MultiCompData/Data/MultiCompData5.npy', help = 'Data directory')
