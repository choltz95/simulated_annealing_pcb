import pyximport; pyximport.install(pyimport=True)
import numpy as np
import sa as simulatedAnnealing
from matplotlib import patches
import matplotlib.pyplot as plt
import collections
import loadYal

from scipy.signal import savgol_filter
import time

from shapely import geometry

def plot_circuit(grid, connections, stats = None):
	board_lower_left_corner = [min(board_dim[0]), min(board_dim[1])]
	width_height = [max(board_dim[0]) - min(board_dim[0]), \
				    max(board_dim[1]) - min(board_dim[1])]

	fig, ax = plt.subplots(1)
	if stats is None:
		ax.set_title(circuit_name)
	else:
		ax.set_title("circuit: " + circuit_name + " wirelength: " + str(round(stats[0],2)) + " overlap: " + str(round(stats[1],2)))
	boundary = patches.Rectangle((min(board_dim[0]), min(board_dim[1])), \
								  max(board_dim[0]) - min(board_dim[0]), max(board_dim[1]) - min(board_dim[1]), \
								  linewidth=1,edgecolor='b',facecolor='none')
	ax.add_patch(boundary)
	for name,shape in grid.items():
		if isinstance(shape, geometry.Point):
			x = shape.x
			y = shape.y
			ax.scatter(x,y, c='b',s=5)
			label = ax.annotate(name, xy=(x, y), fontsize=5, ha="right", va="top", )
		else:
			x, y = shape.exterior.coords.xy
			c = shape.centroid
			points = np.array([x, y], np.int32).T
			polygon_shape = patches.Polygon(points, linewidth=1, edgecolor='r', facecolor='none')
			ax.add_patch(polygon_shape)
			label = ax.annotate(name, xy=(c.x, c.y), fontsize=5, ha="right", va="top")

	for con in connections:
		cc = grid[con].centroid
		for connected_component in connections[con]:
			#if connected_component not in grid or np.isclose(connections[con][connected_component],0):
			if connected_component not in grid:
				continue
			ccc = grid[connected_component].centroid
			ax.plot([cc.x,ccc.x],[cc.y,ccc.y], color="blue", linewidth=1, alpha=0.5, linestyle='dashed')
	plt.show()

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

fname = 'apte.new'
circuit_name = fname.split('.')[0]
data = loadYal.load_yal(fname)
board_dim, components, static_components, connections = get_components(data)

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
adj = [[int(s) for s in row.split()] for row in adj]

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
'M7':{'dimensions':[1.625000, 149.136000]}, 'M8':{'dimensions':[1.625000, 149.136000]}, 'M9':{'dimensions':[1.625000, 149.136000]}, 'BLOCKAGE':{'dimensions':[200, 100]},
'NORTH':{'dimensions':[0,270.72]},'SOUTH':{'dimensions':[0,-270.72]},'EAST':{'dimensions':[732.48,0]},'WEST':{'dimensions':[-732.48,0]}}
board_dim = [[0.0,0.0], [0,270.72],[732.48,270.72],[732.48,0]]
print(board_dim)

components = collections.OrderedDict()
for module in d:
	if 'M' in module:
		components[module] = geometry.Polygon([[0,0],[0, 149.136000],[34.965000, 149.136000],[34.965000, 0]])
	elif 'BLOCKAGE' in module:
		components[module] = geometry.Polygon([[-100,-50], [-100,50], [100,50], [100,-50]])
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
	if i == 9:
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
		if j == 9:
			module2 = 'EAST'
		elif j == 10:
			module2 = 'NORTH'
		elif j == 11:
			module2 = 'SOUTH'
		elif j == 12:
			module2 = 'WEST'

		connections[module1][module2] = int(weight)
"""

sa_start = time.time()
grid, cost, storedCost, stats = simulatedAnnealing.annealing(components, \
															 connections, \
															 static_components, \
															 board_dim)
sa_end = time.time()

print("Time (s)", sa_end - sa_start)
print("Iterations", len(storedCost))
print("Final Cost", cost)
print('Final Wirelength:', stats[0])
print('Final Overlap:', stats[1])

#smoothed_cost = savgol_filter(storedCost, 51, 3)
plt.plot(storedCost)
#plt.xscale('log')
plt.title("Smoothed cost vs interation")
plt.xlabel('interation')
plt.ylabel('Smoothed cost')
plt.show()

plot_circuit(grid, connections, stats)