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
							print(c)
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

fname = 'apte.new'
circuit_name = fname.split('.')[0]
data = loadYal.load_yal(fname)
board_dim, components, static_components, connections = get_components(data)

print(board_dim)
simulatedAnnealing.explore_cost(components, static_components, connections, board_dim)
