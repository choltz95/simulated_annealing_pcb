import pyximport; pyximport.install(pyimport=True)
import numpy as np
import sa as simulatedAnnealing
from matplotlib import patches
import matplotlib.pyplot as plt
import collections
import loadYal
import load_bookshelf
import utils

from scipy.signal import savgol_filter
import time

from shapely import geometry

def get_components(yal_data):
	components = collections.OrderedDict()
	static_components = set()
	connections = {}
	# get board module and components dimensions
	for key in data:
		if 'type' in data[key]:
			if data[key]['type'] == 'PARENT':
				board_dim = data[key]['dimensions']

				for pin in data[key]['iolist']:
					l = pin.split()
					n = l[0]
					x = int(l[2])
					y = int(l[3])
					point = geometry.Point([x,y])
					components[n] = point
					static_components.add(n)

				for con in data[key]['network']:
					l = con.split()
					l.pop(0)
					module = l.pop(0)
					connections[module] = []
					for c in l:
						if c not in components and c not in static_components:
							randpx = np.random.randint(-500 + 50, 10000 - 50)
							randpy = np.random.randint(-500 + 50, 10000 - 50)
							point = geometry.Point([randpx,randpy])
							components[c] = point
						connections[module].append(c)

			elif data[key]['type'] == 'GENERAL':
				dim = data[key]['dimensions']
				coords = [dim[i:i+2] for i in range(0, len(dim), 2)]
				coordslist = [[int(p[0]),int(p[1])] for p in coords]
				cd = [[int(p[0]) for p in coordslist], [int(p[1]) for p in coordslist]]
				width = max(cd[0]) - min(cd[0])
				height = max(cd[1]) - min(cd[1])
				#poly = geometry.Polygon(coordslist)
				randpx = np.random.randint(-500 + 50, 10000 - 50)
				randpy = np.random.randint(-500 + 50, 10000 - 50)
				poly = geometry.Polygon([[randpx,randpy], \
										 [randpx,randpy+height], \
										 [randpx+width, randpy+height], \
										 [randpx+width,randpy]])
				components[key] = poly

	board_dim = [board_dim[x:x+2] for x in range(0, len(board_dim), 2)]
	board_dim = [[int(p[0]) for p in board_dim], [int(p[1]) for p in board_dim]]
	return board_dim, components, static_components, connections

#fname = 'apte.new'
#circuit_name = fname.split('.')[0]
#data = loadYal.load_yal(fname)
#board_dim, components, static_components, connections = get_components(data)

benchmark_dir = 'benchmarks/'
set_dir = 'mcnc/'
cname = 'apte'
tp = './' + benchmark_dir + set_dir + cname
components = load_bookshelf.read_blocks(tp+'.blocks')
placed_components, board_pins = load_bookshelf.read_pl(tp+'.pl')
nets = load_bookshelf.read_nets(tp+'.nets', components, board_pins)
board_pincoords = [(pin[1].x,pin[1].y) for pin in board_pins.items()]
board_pinx, board_piny = zip(*board_pincoords)
board_dim = [max(board_pinx), max(board_piny)] # recover dimension of board from board-pin locations

print(board_dim)
"""
circuit_name = 'coyote'
adj = ["0.000 39714.000 39714.000 39714.000 6660.000 5812.000 5812.000 5812.000 5812.000 0.000 34140.000 1175.000 0.000",
"39714.000 0.000 39714.000 39714.000 6660.000 5812.000 5812.000 5812.000 5812.000 0.000 34140.000 1175.000 0.000", 
"39714.000 39714.000 0.000 39714.000 6660.000 5812.000 5812.000 5812.000 5812.000 0.000 34140.000 1175.000 0.000",
"39714.000 39714.000 39714.000 0.000 6660.000 5812.000 5812.000 5812.000 5812.000 0.000 34140.000 1175.000 0.000", 
"6708.000 6708.000 6708.000 6708.000 0.000 9354.000 9354.000 9354.000 9354.000 0.000 6010.000 299.000 0.000", 
"5812.000 5812.000 5812.000 5812.000 8858.000 0.000 11204.000 10692.000 10180.000 0.000 5144.000 315.000 0.000", 
"5812.000 5812.000 5812.000 5812.000 8858.000 11204.000 0.000 10180.000 10692.000 0.000 5144.000 315.000 0.000", 
"5812.000 5812.000 5812.000 5812.000 8858.000 10692.000 10180.000 0.000 11204.000 0.000 5144.000 315.000 0.000", 
"5812.000 5812.000 5812.000 5812.000 8858.000 10180.000 10692.000 11204.000 0.000 0.000 5144.000 315.000 0.000", 
"0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000", 
"0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000", 
"855.000 855.000 855.000 855.000 299.000 315.000 315.000 315.000 315.000 0.000 0.000 0.000 0.000", 
"0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000"] 
adj = [[float(s) for s in row.split()] for row in adj]

import numpy as np
adj = np.array(adj)
adj = (adj - adj.min()) / (adj.max() - adj.min())
adj = adj.tolist()

static_components = set()
static_components.add('BLOCKAGE')
static_components.add('NORTH')
static_components.add('SOUTH')
static_components.add('EAST')
static_components.add('WEST')

d = {'M1':{'dimensions':[34.965000, 149.136000]}, 'M2':{'dimensions':[34.965000, 149.136000]}, 'M3':{'dimensions':[34.965000, 149.136000]}, 
'M4':{'dimensions':[34.965000, 149.136000]}, 'M5':{'dimensions':[24.345000, 108.816000]}, 'M6':{'dimensions':[41.625000, 149.136000]}, 
'M7':{'dimensions':[41.625000, 149.136000]}, 'M8':{'dimensions':[41.625000, 149.136000]}, 'M9':{'dimensions':[41.625000, 149.136000]}, 

'BLOCKAGE':{'dimensions':[200, 100]},
'NORTH':{'dimensions':[732.48/2,270.72]},'SOUTH':{'dimensions':[732.48/2,0]},'EAST':{'dimensions':[732.48,270.72/2]},'WEST':{'dimensions':[0,270.72/2]}}
#board_dim = [[0.0,0.0], [0,270.72],[732.48,270.72],[732.48,0]]
board_dim = [[0,0,732,732],[0,271,271,0]]
print(board_dim)
"""
"""
circuit_name = "ca_53"

adj = ['0.000 5102.000 7822.000 5102.000 7822.000 5102.000 7822.000 5102.000 16723.000 3933.000 3933.000 3933.000 3933.000 121337.000 121337.000 121337.000 121337.000 7614.000 0.000 1269.000 1269.000 0.000 11865.000 15228.000 0.000 0.000', 
    '4994.000 0.000 4994.000 7854.000 4994.000 7854.000 4994.000 7854.000 16680.000 3725.000 3725.000 3725.000 3725.000 121287.000 121287.000 121287.000 121287.000 7631.000 17.000 1286.000 1286.000 17.000 11535.000 15262.000 0.000 0.000', 
    '7822.000 5102.000 0.000 5102.000 7822.000 5102.000 7822.000 5102.000 16723.000 3933.000 3933.000 3933.000 3933.000 121337.000 121337.000 121337.000 121337.000 7614.000 0.000 1269.000 1269.000 0.000 11865.000 15228.000 0.000 0.000', 
    '4977.000 7837.000 4977.000 0.000 4977.000 7837.000 4977.000 7837.000 16663.000 3708.000 3708.000 3708.000 3708.000 121270.000 121270.000 121270.000 121270.000 7614.000 0.000 1269.000 1269.000 0.000 11535.000 15228.000 0.000 0.000', 
    '7822.000 5102.000 7822.000 5102.000 0.000 5102.000 7822.000 5102.000 16723.000 3933.000 3933.000 3933.000 3933.000 121337.000 121337.000 121337.000 121337.000 7614.000 0.000 1269.000 1269.000 0.000 11865.000 15228.000 0.000 0.000', 
    '4977.000 7837.000 4977.000 7837.000 4977.000 0.000 4977.000 7837.000 16663.000 3708.000 3708.000 3708.000 3708.000 121270.000 121270.000 121270.000 121270.000 7614.000 0.000 1269.000 1269.000 0.000 11535.000 15228.000 0.000 0.000', 
    '7822.000 5102.000 7822.000 5102.000 7822.000 5102.000 0.000 5102.000 16723.000 3933.000 3933.000 3933.000 3933.000 121337.000 121337.000 121337.000 121337.000 7614.000 0.000 1269.000 1269.000 0.000 11865.000 15228.000 0.000 0.000', 
    '4977.000 7837.000 4977.000 7837.000 4977.000 7837.000 4977.000 0.000 16663.000 3708.000 3708.000 3708.000 3708.000 121270.000 121270.000 121270.000 121270.000 7614.000 0.000 1269.000 1269.000 0.000 11535.000 15228.000 0.000 0.000', 
    '2854.000 2854.000 2854.000 2854.000 2854.000 2854.000 2854.000 2854.000 0.000 1356.000 1356.000 1356.000 1356.000 5720.000 5720.000 5720.000 5720.000 120.000 12.000 30.000 30.000 12.000 234.000 240.000 0.000 0.000', 
    '237687.000 237687.000 237687.000 237687.000 237687.000 237687.000 237687.000 237687.000 319862.000 0.000 124497.000 124497.000 124497.000 2080355.000 2080355.000 2080355.000 2080355.000 90774.000 0.000 15129.000 15129.000 0.000 126618.000 181548.000 0.000 0.000',
    '237687.000 237687.000 237687.000 237687.000 237687.000 237687.000 237687.000 237687.000 319862.000 124497.000 0.000 124497.000 124497.000 2080355.000 2080355.000 2080355.000 2080355.000 90774.000 0.000 15129.000 15129.000 0.000 126618.000 181548.000 0.000 0.000', 
    '237687.000 237687.000 237687.000 237687.000 237687.000 237687.000 237687.000 237687.000 319862.000 124497.000 124497.000 0.000 124497.000 2080355.000 2080355.000 2080355.000 2080355.000 90774.000 0.000 15129.000 15129.000 0.000 126618.000 181548.000 0.000 0.000', 
    '237687.000 237687.000 237687.000 237687.000 237687.000 237687.000 237687.000 237687.000 319862.000 124497.000 124497.000 124497.000 0.000 2080355.000 2080355.000 2080355.000 2080355.000 90774.000 0.000 15129.000 15129.000 0.000 126618.000 181548.000 0.000 0.000', 
    '128790.000 128790.000 128790.000 128790.000 128790.000 128790.000 128790.000 128790.000 402861.000 71708.000 71708.000 71708.000 71708.000 0.000 3643707.000 3643707.000 3643707.000 224802.000 0.000 35829.000 35829.000 0.000 137583.000 449604.000 0.000 0.000', 
    '128790.000 128790.000 128790.000 128790.000 128790.000 128790.000 128790.000 128790.000 402861.000 71708.000 71708.000 71708.000 71708.000 3643707.000 0.000 3643707.000 3643707.000 224802.000 0.000 35829.000 35829.000 0.000 137583.000 449604.000 0.000 0.000', 
    '116742.000 116742.000 116742.000 116742.000 116742.000 116742.000 116742.000 116742.000 389601.000 62720.000 62720.000 62720.000 62720.000 3641691.000 3641691.000 0.000 3641691.000 177552.000 0.000 27954.000 27954.000 0.000 137583.000 355104.000 0.000 0.000', 
    '128790.000 128790.000 128790.000 128790.000 128790.000 128790.000 128790.000 128790.000 402861.000 71708.000 71708.000 71708.000 71708.000 3643707.000 3643707.000 3643707.000 0.000 224802.000 0.000 35829.000 35829.000 0.000 137583.000 449604.000 0.000 0.000', 
    '0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 368954.000 35567.000 35567.000 368954.000 0.000 211592.000 0.000 0.000', 
    '0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 9840.000 9840.000 9840.000 9840.000 133280.000 0.000 31120.000 31120.000 313278.000 0.000 266560.000 0.000 0.000', 
    '1286.000 1286.000 1286.000 1286.000 1286.000 1286.000 1286.000 1286.000 6231.000 1004.000 1004.000 1004.000 1004.000 42657.000 42657.000 42657.000 42657.000 2664968.000 8591212.000 0.000 877393.000 8591212.000 5901.000 5331424.000 0.000 0.000',
    '1286.000 1286.000 1286.000 1286.000 1286.000 1286.000 1286.000 1286.000 6231.000 1004.000 1004.000 1004.000 1004.000 42657.000 42657.000 42657.000 42657.000 2610532.000 8431500.000 860436.000 0.000 8431500.000 5901.000 5222552.000 0.000 0.000', 
    '0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 984.000 984.000 984.000 984.000 13328.000 31223.000 3112.000 3112.000 0.000 0.000 26656.000 0.000 0.000', 
    '98008.000 98118.000 98008.000 98118.000 98008.000 98118.000 98008.000 98118.000 145512.000 49147.000 49147.000 49147.000 49147.000 654272.000 654272.000 654272.000 654272.000 227790.000 307753.000 60415.000 60415.000 307234.000 0.000 0.000 0.000 0.000', 
    '0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000', 
    '0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000', 
    '0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000']
adj = [[int(float(s)) for s in row.split()] for row in adj]

import numpy as np
adj = np.array(adj)
adj = (adj - adj.min()) / (adj.max() - adj.min())
adj = adj.tolist()

static_components = set()
static_components.add('BLOCKAGE')
static_components.add('NORTH')
static_components.add('SOUTH')
static_components.add('EAST')
static_components.add('WEST')

d = {'M1':{'dimensions':[64.440000, 98.736000]}, 'M2':{'dimensions':[64.440000, 98.736000]}, 'M3':{'dimensions':[64.440000, 98.736000]}, 
'M4':{'dimensions':[64.440000, 98.736000]}, 'M5':{'dimensions':[64.440000, 98.736000]}, 'M6':{'dimensions':[64.440000, 98.736000]}, 
'M7':{'dimensions':[64.440000, 98.736000]}, 'M8':{'dimensions':[64.440000, 98.736000]}, 'M9':{'dimensions':[39.690000, 28.944000]},
'M10':{'dimensions':[26.460000, 49.104000]}, 'M11':{'dimensions':[26.460000, 49.104000]}, 'M12':{'dimensions':[26.460000, 49.104000]},
'M13':{'dimensions':[26.460000 ,49.104000]}, 'M14':{'dimensions':[29.340000, 139.056000]}, 'M15':{'dimensions':[29.340000, 139.056000]},
'M16':{'dimensions':[29.340000, 139.056000]}, 'M17':{'dimensions':[29.340000, 139.056000]}, 'M18':{'dimensions':[110.520000, 104.784000]},
'M19':{'dimensions':[26.460000 ,80.592000]}, 'M20':{'dimensions':[37.980000, 49.104000]}, 'M21':{'dimensions':[37.980000, 49.104000]},
'M22':{'dimensions':[26.460000 ,70.512000]},

'BLOCKAGE':{'dimensions':[500, 400]}, 'NORTH':{'dimensions':[820/2,738]},'SOUTH':{'dimensions':[820/2,0]},'EAST':{'dimensions':[820,738/2]},'WEST':{'dimensions':[0,738/2]}}

board_dim = [[0,0,820,820],[0,738,738,0]]
print(board_dim)
"""
"""
components = collections.OrderedDict()
for module in d:
	if 'M' in module:
		#components[module] = geometry.Polygon([[0,0],[0, 149.136000],[34.965000, 149.136000],[34.965000, 0]])
		dim = d[module]['dimensions']
		randx = np.random.randint(0,750)
		randy = np.random.randint(0,650)
		components[module] = geometry.Polygon([[randx,randy],[randx+dim[0],randy],[randx+dim[0],randy+dim[1]],[randx,randy+dim[1]]])
	elif 'BLOCKAGE' in module:
		components[module] = geometry.Polygon([[0,0],[0,100],[200,100],[200,0]]) #[[0,0],[0,400],[500,400],[500,0]]
	elif 'EAST' in module:
		components[module] = geometry.Point(d[module]['dimensions'])
	elif 'NORTH' in module:
		components[module] = geometry.Point(d[module]['dimensions'])
	elif 'SOUTH' in module:
		components[module] = geometry.Point(d[module]['dimensions'])
	elif 'WEST' in module:
		components[module] = geometry.Point(d[module]['dimensions'])

connections = {}
for i,row in enumerate(adj):
	module1 = 'M' + str(i+1)
	if i == 9: #22
		module1 = 'EAST'
	elif i == 10:
		module1 = 'NORTH'
	elif i == 11:
		module1 = 'SOUTH'
	elif i == 12:
		module1 = 'WEST'
	connections[module1] = {}
	for j, weight in enumerate(row):
		module2 = 'M' + str(j+1)
		if j == 9: #22
			module2 = 'EAST'
		elif j == 10:
			module2 = 'NORTH'
		elif j == 11:
			module2 = 'SOUTH'
		elif j == 12:
			module2 = 'WEST'

		connections[module1][module2] = float(weight)
"""
sa_start = time.time()
#grid, cost, storedCost, stats = simulatedAnnealing.multistart(components, 
#															  nets, 
#															  board_pins, 
#															  board_dim,
#															  simulatedAnnealing.cost)
board_dim = [[0,board_dim[0],board_dim[0],0],[0,0,board_dim[1],board_dim[1]]]
blocks, cost, storedCost, stats, T, idx = simulatedAnnealing.annealing(components, 
															  nets, 
															  board_pins, 
															  board_dim,
															  0, 10000000,
															  simulatedAnnealing.cost)
sa_end = time.time()

print("Time (s)", sa_end - sa_start)
print("Iterations", len(storedCost))
print("Final Cost", cost)
print('Final Wirelength:', stats[0])
print('Final Overlap:', stats[1])

smoothed_cost = savgol_filter(storedCost, 51, 3)
plt.plot(smoothed_cost)
plt.title("Smoothed cost vs interation")
plt.xlabel('interation')
plt.ylabel('Smoothed cost')
plt.show()

utils.plot_circuit(cname,components, nets, board_dim, stats)