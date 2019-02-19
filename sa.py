import random, math
import numpy as np

import copy, sys, time
import collections

import shapely
from shapely.ops import cascaded_union
from shapely.affinity import translate, rotate
from rtree import index
from rtree.index import Rtree
from tqdm import tqdm

import multiprocessing as mp
try:
    cpus = mp.cpu_count()
except NotImplementedError:
    cpus = 2
       
INF = 100000
ALPHA = 0.96
ALPHA_INCREASING = False

L1 = 0.99 # wirelength parameter
L2 = 0.01 # overlap parameter

def fix_boundary_constraints(grid, element, board_dim):
	"""
	Computes the minimum transformation for a modules 
	to satisfy board boundary positional constraints
	:param grid: dict - board representation
	:param elements: [polygon] - polygon(s) to be translated
	:param board_dim: int in (0.0, 1.0) - the target probability
	:return: polygon
	"""
	xs,ys = grid[element].exterior.xy
	halfwidth = 0.5 * abs(max(xs) - min(xs))
	halfheight = 0.5 * abs(max(ys) - min(ys))
	centroid = grid[element].centroid
	cx = centroid.x
	cy = centroid.y

	if cx + halfwidth > max(board_dim[0]):
		grid[element] = translate(grid[element], -abs(cx + halfwidth - max(board_dim[0])),0)
	elif cx - halfwidth < min(board_dim[0]):
		grid[element] = translate(grid[element], abs(cx - halfwidth - min(board_dim[0])),0)

	if cy + halfheight > max(board_dim[1]):
		grid[element] = translate(grid[element], 0, -abs(cy + halfheight - max(board_dim[1])))
	elif cy - halfheight < min(board_dim[1]):
		grid[element] = translate(grid[element], 0, abs(cy - halfheight - min(board_dim[1])))

def transition(grid, 
		   	   idx, 
		   	   board_dim, 
		   	   connections, 
		       static_components, 
		       current_cost, 
		       stats,
		       l):
	"""
	Updates the state of the board 
	:param grid: dict - board representation
	:param idx: 
	:param board_dim: int in (0.0, 1.0) - the target probability
	:param connections:
	:param static_components:
	:param cost:
	:param stats:
	:param l:
	:return: grid
	"""
	altered = []
	intersections = []

	element1 = random.choice(list(grid))
	element2 = random.choice(list(grid))
	c1 = grid[element1].centroid
	c2 = grid[element2].centroid
	c1x = c1.x 
	c1y = c1.x 
	c2x = c2.x 
	c2y = c2.y 

	while element1 in static_components:
		element1= random.choice(list(grid))
	while element2==element1 or element2 in static_components:
		element2 = random.choice(list(grid))
	altered_modules = [element1, element2]

	updated_modules = [[(element1, grid[element1]),(element2, grid[element2])]]

	i1 = list(grid).index(element1)
	i2 = list(grid).index(element2)

	initial_cost, initial_stats = cost(altered_modules, grid, idx, connections, l)
	roll = np.random.random()
	if(roll<0.8): # swap two modules
		xtransform = c1x - c2x
		ytransform = c1y - c2y

		idx.delete(i1, grid[element1].bounds)
		idx.delete(i2, grid[element2].bounds)

		grid[element1] = translate(grid[element1], -xtransform, -ytransform)
		grid[element2] = translate(grid[element2], xtransform, ytransform)
		fix_boundary_constraints(grid, element1, board_dim)
		fix_boundary_constraints(grid, element2, board_dim)

		idx.insert(i1, grid[element1].bounds)
		idx.insert(i2, grid[element2].bounds)
	elif(roll<0.4): # shift a module
		xs = grid[element1].exterior.xy[0] 
		ys = grid[element1].exterior.xy[1] 
		halfwidth = 0.5 * abs(max(xs) - min(xs))
		halfheight = 0.5 * abs(max(ys) - min(ys))
		randpx = np.random.randint(min(board_dim[0]) + halfwidth,
								   max(board_dim[0]) - halfwidth)
		randpy = np.random.randint(min(board_dim[1]) + halfheight,
								   max(board_dim[1]) - halfheight)
		randp = [randpx,randpy]

		#c1 = grid[element1].centroid
		xtransform = c1x - randp[0]
		ytransform = c1y - randp[1]
		idx.delete(i1, grid[element1].bounds)
		grid[element1] = translate(grid[element1], -xtransform, -ytransform)
		idx.insert(i1, grid[element1].bounds)
	else: # rotate a module
		randp = random.choice([90,180,270])
		idx.delete(i1, grid[element1].bounds)
		grid[element1] = rotate(grid[element1], randp)
		fix_boundary_constraints(grid, element1, board_dim)
		idx.insert(i1, grid[element1].bounds)

	updated_modules.append([(element1, grid[element1]),\
							(element2, grid[element2])])
	transition_cost, transition_stats = cost(altered_modules, grid, idx, connections, l)
	updated_cost = current_cost - initial_cost + transition_cost
	updated_stats = [stats[0] - initial_stats[0] + transition_stats[0], \
					 stats[1] - initial_stats[1] + transition_stats[1]]
	#updated_cost, updated_stats = cost(list(grid),grid,idx,connections)

	return grid, idx, updated_cost, updated_stats, updated_modules

def deupdate(grid, 
			 idx, 
			 altered_modules, 
			 update):
	"""
	remove a set of items from idx. 
	:param grid: dict - board representation
	:param idx: 
	:param altered_modules
	:param update
	"""
	accepted_modules = altered_modules[1-update]
	rejected_modules = altered_modules[update]

	for accepted_module, rejected_module in zip(accepted_modules,rejected_modules):
		module_id = list(grid).index(accepted_module[0])
		accepted_module = accepted_module[1]
		rejected_module = rejected_module[1]
		amb = accepted_module.bounds
		rmb = rejected_module.bounds

		if update == 0: # rejected id is true id
			idx.delete(module_id,amb)
			idx.delete(module_id,rmb)
			idx.insert(module_id,amb)
			key = list(grid.keys())[module_id]
			grid[key] = accepted_module
		elif update == 1: # accepted is true id
			idx.delete(module_id,rmb)
			idx.delete(module_id,accepted_module.bounds)
			idx.insert(module_id,amb)
			key = list(grid.keys())[module_id]
			grid[key] = accepted_module

def update_parameters(T,
				 	  l1,
				 	  l2, 
				 	  alpha, 
				 	  alpha_increasing):
	"""
	Updates iterate-dependent parameters
	:param T
	:param l1
	:param l2
	:param alpha
	:param alpha_increasing
	:return: list
	"""
	"""
	if alpha_increasing == True:
		alpha = alpha + 0.001
	if alpha_increasing == False:
		alpha = alpha - 0.001

	if alpha >= 0.95 and alpha_increasing == True:
		alpha_increasing = False
	if alpha <= 0.8 and alpha_increasing == False:
		alpha_increasing = True
	"""
	if l1 > 0.1:
		l1 = l1 - 0.00085
		l2 = l2 + 0.00085
	T = alpha*T
	return T,l1,l2,alpha,alpha_increasing

def intersection_area(modules, grid, idx):
	"""
	Compute intersectional area of all placed components
	:param modules
	:param grid
	:param idx
	:return: float
	"""
	intersectional_area = 0
	module_history = set()
	for m in modules:
		module_history.add(m)
		for intersection in [list(grid.values())[pos] for pos in idx.intersection(grid[m].bounds) if list(grid.keys())[pos] not in module_history]:
			intersectional_area += grid[m].intersection(intersection).area
	return intersectional_area

def wirelength(modules, 
			   grid, 
			   connections, 
			   is_coyote=False):
	"""
	Compute cumulative wirelength
	:param modules:
	:param grid:
	:param connections:
	:return: float
	"""
	if is_coyote:
		wirelength = 0
		module_history = set()
		for module in modules:
			if module not in connections:
				continue
			module_history.add(module)
			centroid = grid[module].centroid
			cx = centroid.x
			cy = centroid.y
			for m in connections[module]:
				if m in grid and m not in module_history:
					connected_module_centroid = grid[m].centroid
					weight = connections[module][m]
					wirelength += weight * np.absolute(connected_module_centroid.x - cx) + \
					  			  			np.absolute(connected_module_centroid.y - cy)
	else:
		wirelength = 0
		module_history = set()
		for module in modules:
			if module not in connections:
				continue
			module_history.add(module)
			centroid = grid[module].centroid
			cx = centroid.x
			cy = centroid.y
			for connected_module in connections[module]:
				if connected_module in grid and connected_module not in module_history:
					connected_module_centroid = grid[connected_module].centroid
					wirelength += np.absolute(connected_module_centroid.x - cx) + \
								  np.absolute(connected_module_centroid.y - cy)
	return wirelength

def cost(modules, 
		 grid, 
		 idx, 
		 connections,
		 l):
	"""
	Compute weighted cost
	:param modules:
	:param grid:
	:param idx:
	:param connections:
	:param l: 
	:return: float
	"""
	l1,l2 = l
	wl = wirelength(modules, grid, connections)
	ia = intersection_area(modules, grid, idx)
	return l1*wl + l2*ia, [wl, ia]

def annealing(grid, 
			  connections, 
			  static_components, 
			  board_dim,
			  seed=100,
			  #costfunc=lambda modules, board_state, r_tree, weights, subcost: 1.0
			  ):
	"""
	Annealing
	:param grid:
	:param connections:
	:param static_components:
	:param board_dim: 
	:return: float
	"""
	random.seed(seed)
	np.random.seed(seed)
	T = INF
	l1 = L1
	l2 = L2
	alpha = ALPHA
	alpha_increasing = ALPHA_INCREASING

	p = index.Property()
	idx = index.Index(properties=p, interleaved=True)
	for i, key in enumerate(grid):
		if key in static_components:
			continue
		idx.insert(i, grid[key].bounds)

	min_cost, min_stats = cost(list(grid), grid, idx, connections,[l1,l2])
	cost_history = [min_cost]
	temp_cost = 1
	temp_stats = min_stats
	for i in tqdm(range(500)):
		for i in range(10):
			tempGrid, idx, temp_cost, temp_stats, updated_modules = transition(grid,
																			   idx, board_dim,
																			   connections, static_components,
																			   min_cost, min_stats,
																			   [l1,l2])
			delta = temp_cost - min_cost
			if (delta<0):
				grid = tempGrid
				min_cost = temp_cost
				min_stats = temp_stats
				cost_history.append(min_cost)
				deupdate(grid, idx, updated_modules, 0)
			else:
				p = np.exp(-delta / T)
				if(np.random.random()<p):
					grid = tempGrid
					min_cost = temp_cost
					min_stats = temp_stats
					cost_history.append(min_cost)
					deupdate(grid, idx, updated_modules, 0)
				else:
					cost_history.append(min_cost)
					deupdate(grid, idx, updated_modules, 1)
		T,l1,l2,alpha,alpha_increasing = update_parameters(T,l1,l2,alpha,alpha_increasing)
	return grid, min_cost, cost_history, min_stats