import argparse 
import numpy as np 

parser = argparse.ArgumentParser(description='Finding homology with active learning')

parser.add_argument('--N', type = int, default = 100, help = 'number of filtering')

parser.add_argument('--C', type = int, default = 2, help = 'Clustering number')

parser.add_argument('--Trial', type = int, default = 1, help = 'Trial number')

parser.add_argument('--S', type = float, default = 0, help = 'Smallest filtering scale')

parser.add_argument('--L', type = float, default = 1.5, help = 'Largest filtering scale')

parser.add_argument('--NT', type = int, default = 0.1, help = 'Noise removal threshold')

parser.add_argument('--K', type = int, default = 15 , help = 'number of considered nearest opposite neighbors')

parser.add_argument('--R', type = float, default = 0.3, help = 'radius of graph for complex construction')

parser.add_argument('--r', type = float, default = 0.3, help = 'radius scale of graph for S2 label query')

parser.add_argument('--LoadQueryIndex', type = int, default = 0, help = 'load query index')

parser.add_argument('--LoadRipser', type = int, default = 0, help = 'load ripsers indicator')

parser.add_argument('--DrawDist', type = int, default = 1, help = 'draw persistance diagrams distance or not')

parser.add_argument('--DrawPD', type = int, default = 1, help = 'draw persistance diagrams or not')

parser.add_argument('--DrawBoundary', type = int, default = 0, help = 'draw boundaries of cutset or not')

parser.add_argument('--Per', type = float, default = 1, help = 'The proportion of used unlabelled data pool')

parser.add_argument('--Graph', type = str, default = 'NN', help = 'Graph type: Radius or NN')

parser.add_argument('--Dist', type = str, default = 'Bottleneck', help = 'ways to compare persistance diagrams: bottleneck, sliced_wasserstein or heat')

parser.add_argument('--DataType', type = str, default = 'Syn', help = 'Data type: Syn or Real')

parser.add_argument('--Comp', type = str, default = 'LVR', help = 'Check complex type: VR or LVR')

