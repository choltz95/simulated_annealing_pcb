import sys
import yaml
import json
import numpy as np
import ast
import utils
import shapely
from shapely.geometry import Point, Polygon

def read_pl(fname):
	"""
	Read & parse .pl (placement) file
	:param fname: .pl filename
	"""
	with open(fname,'r') as f:
		lines = f.read().splitlines()

	t = -1
	if 'UMICH' in lines[0]:
		t = 0 # generally inp file
	elif 'UCLA' in lines[0]:
		t = 1 # generally parquetfp output type
	else:
		print('.pl file parse error')
		return 0
	lines = lines[4:]
	components = {}
	board_pins = {}
	bp = 0

	if t == 0:
		for line in lines:
			if line == '':
				bp = 1
				continue
			if bp == 0:
				l = line.split()
				cname = l[0]
				dims = l[5]+l[6]
				dims = [float(i) for i in dims[1:-1].split(",")]
				x = int(l[1])
				y = int(l[2])
				poly = Polygon([[x,y], \
							   [x + dims[0],y], \
							   [x + dims[0],y+dims[1]], \
							   [x,y+dims[1]]])
				components[cname] = poly
			else:
				l = line.split()
				pname = l[0]
				coords = Point([int(l[1]),int(l[2])])
				board_pins[pname] = coords
	elif t == 1:
		for line in lines:
			if line == '':
				bp = 1
				continue
			if bp == 0:
				components = None
				pass # .blks should handle block placement
			else:
				l = line.split()
				pname = l[0]
				coords = Point([int(l[1]),int(l[2])])
				board_pins[pname] = coords

	return components, board_pins

def read_nets(fname,components,board_pins):
	"""
	Read & parse .nets (netlist) file
	:param fname: .nets filename
	"""
	nets = []
	pin2comp = {} # {pin : component}
	comp2pin = {} # {component : [pins]}
	i = -1
	with open(fname, 'r') as f:
		lines = f.read().splitlines()[8:]
	for line in lines:
		if '#' in line:
			continue
		if 'NetDegree' in line:
			i += 1
			nets.append([])
			continue

		l = line.split()
		if len(l) < 3:
			pin_name = l[0]
			pin = board_pins[pin_name]
			nets[i].append([pin_name,pin])
		else:
			pin_name = l[0]
			local_pin_loc = [float(l[3][1:])/100.0 + 0.5, float(l[4][1:])/100.0 + 0.5]
			#pinx,piny = utils.pin_pos([pin_name, local_pin_loc], components)
			#print(pin_name, local_pin_loc, pinx, piny)
			#pin = Point([pinx,piny])
			pin = local_pin_loc
			nets[i].append([pin_name, pin])

	return nets

def read_blocks(fname):
	"""
	Read & parse .blk (blocks) file
	:param fname: .blk filename
	"""
	blocks = {}
	with open(fname, 'r') as f:
		lines = f.read().splitlines()[9:]
	components = {}
	bp = 0
	for line in lines:
		if line == '':
			bp = 1
			continue
		if bp == 0:
			l = line.split()
			cname = l[0]
			vstring =  ' '.join(l[3:])
			vstring = '[' + vstring.replace(') (', '),(') + ']'
			vertices = ast.literal_eval(vstring)
			poly = Polygon(vertices)
			components[cname] = poly
		#else:
		#	l = line.split()
		#	pname = l[0]
		#	print(l)
		#	coords = Point([int(l[1]),int(l[2])])
		#	board_pins[pname] = coords

	return components

def write_pl(fname):
	pass

#plfile = sys.argv[1]
#netsfile = sys.argv[2]
#blocksfile = sys.argv[3]
"""
components:        dictionary of placed component-polygons indexed by their name
placed_components: dictionary of placed component-polygons indexed by their name
board_pins:        dictionary of static point-pins indexed by their name
nets:              list of k nets, each net is a list of n point-pins
"""
#components = read_blocks(blocksfile)
#placed_components, board_pins = read_pl(plfile)
#nets = read_nets(netsfile, components, board_pins)

#print('pcb data loaded')