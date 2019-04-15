from matplotlib import patches
import matplotlib.pyplot as plt
import sys

import shapely
from shapely import geometry

import numpy as np

import load_bookshelf

def plot_circuit(circuit_name, components, nets, board_dim, stats = None):
	"""
	board dim: [[xs],[ys]]
	"""
	board_lower_left_corner = [min(board_dim[0]), min(board_dim[1])]
	width_height = [max(board_dim[0]) - min(board_dim[0]), \
				    max(board_dim[1]) - min(board_dim[1])]

	fig, ax = plt.subplots(1)
	if stats is None:
		ax.set_title(circuit_name)
	else:
		ax.set_title("circuit: " + circuit_name+ " wirelength: " + str(round(stats[0],2)) + " overlap: " + str(round(stats[1],2)))
	boundary = patches.Rectangle((min(board_dim[0]), min(board_dim[1])), \
								  max(board_dim[0]) - min(board_dim[0]), max(board_dim[1]) - min(board_dim[1]), \
								  linewidth=1,edgecolor='b',facecolor='none')
	ax.add_patch(boundary)
	for name,shape in components.items():
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

	# visualize nets
	for net in nets:
		netlist = []
		c = [np.random.rand(3)] # random color
		for pin in net:
			if pin[0] not in components:
				cx = pin[1].x
				cy = pin[1].y
			else:
				cx, cy = pin_pos(pin,components)
			netlist.append([cx,cy])
		xmax = max([p[0] for p in netlist])
		xmin = min([p[0] for p in netlist])
		ymax = max([p[1] for p in netlist])
		ymin = min([p[1] for p in netlist])
		center =  [(xmax + xmin)/2,(ymax + ymin)/2]
		for i in range(len(netlist)):
			ax.plot([netlist[i][0],center[0]],[netlist[i][1],center[1]], color=tuple(map(tuple, c))[0] + (255,), linewidth=1, alpha=0.25, linestyle='dashed')
		xs= [ x[0] for x in netlist ]
		ys= [ x[1] for x in netlist ]
		ax.scatter(xs,ys,marker='.',c=c)
		ax.scatter(center[0],center[1],marker='.',c=c)
	plt.xlim(-50, max(width_height) + 50)
	plt.ylim(-50, max(width_height) + 50)
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()

def pin_pos(pin_loc, modules):
	"""
	Convert localized pin positions to position wrt
	 global coordinates
	:param pin_loc: pin location of the form [pinname, [%x, %y]]
	:param modules: list of modules
	"""
	module_name, local_pin_loc = pin_loc
	minx, miny, maxx, maxy = modules[module_name].bounds

	pinx = (maxx - minx) * local_pin_loc[0] + minx
	piny = (maxy - miny) * local_pin_loc[1] + miny
	return pinx, piny

plfile = sys.argv[1]
components,board_pins = load_bookshelf.read_pl(plfile)
netsfile = sys.argv[2]
nets,mod2net = load_bookshelf.read_nets(netsfile,components,board_pins)
if board_pins is not None and len(board_pins) > 0:
	xs = [pin[1].x for pin in board_pins.items()]
	ys = [pin[1].y for pin in board_pins.items()]
	board_dim = [xs,ys]
else:
	board_dim = [500,500]
plot_circuit(plfile.split('.')[0], components,nets,board_dim)
