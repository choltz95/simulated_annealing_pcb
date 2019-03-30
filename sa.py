import random, math
import numpy as np
import numba
from numba import jit

import copy, sys, time
import collections
from operator import itemgetter

import shapely
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union
from shapely.affinity import translate, rotate
from rtree import index
from rtree.index import Rtree
from tqdm import tqdm
import itertools

import utils

import multiprocessing as mp
try:
    num_cpus = mp.cpu_count()
except NotImplementedError:
    num_cpus = 2
       
INF = 100000
ALPHA = 0.995
ALPHA_INCREASING = False

L1_eval = 0.4
L2_eval = 0.6

L1 = 0.99 # wirelength parameter
L2 = 0.01 # overlap parameter

def gen_random_grid(grid,board_dim,static_components):
	for element in grid:
		if element in static_components:
			continue
		try:
			xs,ys = grid[element].exterior.xy
			halfwidth = 0.5 * abs(max(xs) - min(xs))
			halfheight = 0.5 * abs(max(ys) - min(ys))
			centroid = grid[element].centroid
			cx = centroid.x
			cy = centroid.y
		except:
			xs = [grid[element].x]
			ys = [grid[element].y]
			halfwidth = 0
			halfheight = 0
			cx = xs[0]
			cy = ys[0]

		randpx = np.random.randint(min(board_dim[0]) + halfwidth,
								   max(board_dim[0]) - halfwidth)
		randpy = np.random.randint(min(board_dim[1]) + halfheight,
								   max(board_dim[1]) - halfheight)
		randp = [randpx,randpy]

		xtransform = cx - randp[0]
		ytransform = cy - randp[1]

		grid[element] = translate(grid[element], -xtransform, -ytransform)
		fix_boundary_constraints(grid, element, board_dim)
	return grid

def explore_cost(grid, 
				  static_components, 
				  connections,
				  board_dim):
	cost_hist = []

	for _ in tqdm(range(100)):
		grid = gen_random_grid(grid,board_dim,static_components)
		c,stats = cost(list(grid), grid, None, connections, mod2net, [L1_eval,L2_eval])
		cost_hist.append((grid,c,stats))

def fix_boundary_constraints(grid, element, board_dim):
	"""
	Computes the minimum transformation for a module's 
	to satisfy board boundary positional constraints
	:param grid: dict - board representation
	:param elements: [Polygon] - polygon(s) to be translated
	:param board_dim: int in (0.0, 1.0) - the target probability
	:return: polygon
	"""
	try:
		minx,miny,maxx,maxy = grid[element].bounds
		halfwidth = 0.5 * (maxx - minx)
		halfheight = 0.5 *(maxy - miny)
		centroid = grid[element].centroid
		cx = centroid.x
		cy = centroid.y
	except:
		xs = [grid[element].x]
		ys = [grid[element].y]
		halfwidth = 0
		halfheight = 0
		cx = xs[0]
		cy = ys[0]

	if cx + halfwidth > max(board_dim[0]):
		grid[element] = translate(grid[element], -(cx + halfwidth - max(board_dim[0])),0)
	elif cx - halfwidth < min(board_dim[0]):
		grid[element] = translate(grid[element], abs(cx - halfwidth - min(board_dim[0])),0)

	if cy + halfheight > max(board_dim[1]):
		grid[element] = translate(grid[element], 0, -(cy + halfheight - max(board_dim[1])))
	elif cy - halfheight < min(board_dim[1]):
		grid[element] = translate(grid[element], 0, abs(cy - halfheight - min(board_dim[1])))

def transition(grid, 
		   	   idx, 
		   	   board_dim, 
		   	   connections, 
		   	   mod2net,
		       static_components, 
		       iteration,
		       current_cost,
		       stats,
		       costfunc,
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

	c1 = grid[element1].centroid
	c1x = c1.x 
	c1y = c1.x 

	altered_modules = [element1]
	updated_modules = [[(element1, grid[element1])]]

	i1 = list(grid).index(element1)

	roll = np.random.random()
	if(roll<0.4): # swap two modules
		element2 = random.choice(list(grid))
		while element2 == element1:
			element2 = random.choice(list(grid))
		c2 = grid[element2].centroid
		c2x = c2.x 
		c2y = c2.y 
		altered_modules.append(element2)
		updated_modules[0].append((element2, grid[element2]))
		i2 = list(grid).index(element2)
		initial_cost, initial_stats = costfunc(altered_modules, grid, idx, connections, mod2net, l)

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
	elif(roll<0.8): # shift a module
		initial_cost, initial_stats = costfunc(altered_modules, grid, idx, connections, mod2net, l)
		try:
			minx,miny,maxx,maxy = grid[element1].bounds
		except: # point
			maxx = [grid[element1].x]
			maxy = [grid[element1].y]
			minx = maxx
			miny = maxy
		halfwidth = 0.5 * (maxx - minx)
		halfheight = 0.5 * (maxy - miny)

		shift_ball = max(0.9955**iteration,5)
		rx = shift_ball*max(board_dim[0])
		ry = shift_ball*max(board_dim[1])
		randpx = np.random.randint(max(min(board_dim[0]) + halfwidth,c1x-rx),
								   min(max(board_dim[0]) - halfwidth,c1x+rx))
		randpy = np.random.randint(max(min(board_dim[1]) + halfheight,c1x-ry),
								   min(max(board_dim[1]) - halfheight,c1x+ry))
		randp = [randpx,randpy]

		xtransform = c1x - randp[0]
		ytransform = c1y - randp[1]

		idx.delete(i1, grid[element1].bounds)
		grid[element1] = translate(grid[element1], -xtransform, -ytransform)
		fix_boundary_constraints(grid, element1, board_dim)
		idx.insert(i1, grid[element1].bounds)
	else: # rotate a module
		initial_cost, initial_stats = costfunc(altered_modules, grid, idx, connections, mod2net, l)
		randp = random.choice([90,180,270])
		idx.delete(i1, grid[element1].bounds)
		grid[element1] = rotate(grid[element1], randp, origin="center")
		fix_boundary_constraints(grid, element1, board_dim)
		idx.insert(i1, grid[element1].bounds)

	if len(altered_modules) > 1:
		updated_modules.append([(element1, grid[element1]),\
								(element2, grid[element2])])
	else:
		updated_modules.append([(element1, grid[element1])])
									
	transition_cost, transition_stats = costfunc(altered_modules, grid, idx, connections, mod2net, l)
	updated_cost = current_cost - initial_cost + transition_cost
	updated_stats = [np.abs(stats[0] - initial_stats[0] + transition_stats[0]), \
					 stats[1] - initial_stats[1] + transition_stats[1]]
	#true_cost = costfunc(list(grid),grid,idx,connections, mod2net,l)
	#print(updated_stats, true_cost[1])
	return grid, idx, updated_cost, updated_stats, updated_modules

#@jit(parallel = True, nogil = True)
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

#@jit(nopython=True)
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
	if l1 > 0.28:
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
	if idx is None:
		print('nyeeeet')
		for m in modules:
			module_history.add(m)
			for mc in list(grid):
				mm = grid[mc]
				intersectional_area += grid[m].intersection(mm).area
	else:
		for m in modules:
			module_history.add(m)
			for mm in [list(grid.items())[pos] for pos in idx.intersection(grid[m].bounds) if list(grid.keys())[pos] not in module_history]:
				if mm[0] in module_history:
					continue 
				intersectional_area += grid[m].intersection(mm[1]).area
	return int(intersectional_area)

def hpwl():
	pass

def euclidean():
	pass

def manhattan():
	pass

#@jit(parallel = True, nogil = True)
def wirelength(modules, 
			   grid, 
			   connections, 
			   mod2net,
			   is_coyote=False):
	"""
	Compute cumulative wirelength
	:param modules:
	:param grid:
	:param connections:
	:return: float
	"""
	nets = connections
	wirelength = 0
	module_history = set()
	hpwl = 0
	for net in set([n for m in modules for n in mod2net[m]]):
		plxs = []
		plys = []
		for pin in nets[net]: # for each pin in connection
			pname = pin[0]
			if pname in grid:
				pinx, piny = utils.pin_pos(pin, grid)
				plxs.append(pinx)
				plys.append(piny)
			else:
				pinx = pin[1].x
				piny = pin[1].y
				plxs.append(pinx)
				plys.append(piny)

		yd = max(plys) - min(plys)
		xd = max(plxs) - min(plxs)
		hpwl += yd + xd
	return int(hpwl)

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
					#wirelength += weight * (np.abs(connected_module_centroid.x - cx) + \
					#  			  			np.abs(connected_module_centroid.y - cy))
					wirelength += weight * np.sqrt(np.square(connected_module_centroid.x - cx) + \
					  			  			np.square(connected_module_centroid.y - cy))
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
					##wirelength += np.abs(connected_module_centroid.x - cx) + \
					##			  np.abs(connected_module_centroid.y - cy)
					wirelength += np.sqrt(np.square(connected_module_centroid.x - cx) + \
  			  							np.square(connected_module_centroid.y - cy))
	return int(wirelength)
	"""

def cost(modules, 
		 grid,  
		 idx, 
		 connections,
		 mod2net,
		 l):
	"""
	Compute weighted cost
	:param modules:
	:param grid:
	:param idx:ngth
	:param connections:
	:param l: 
	:return: float
	"""
	l1,l2 = l
	wl = wirelength(modules, grid, connections, mod2net)
	ia = intersection_area(modules, grid, idx)
	return l1*wl + l2*ia, [wl, ia]

def annealing(blocks, 
			  nets, 
			  mod2net,
			  board_pins, 
			  board_dim,
			  pos,
			  T_0,
			  costfunc,
			  seed=None
			  #costfunc=lambda modules, board_state, r_tree, weights, subcost: 1.0
			  ):
	"""
	Annealing
	:param blocks:
	:param nets:
	:param board_pins:
	:param board_dim: 
	:return: float
	"""
	random.seed(seed)
	np.random.seed(seed)
	T = T_0
	l1 = L1
	l2 = L2
	alpha = ALPHA
	alpha_increasing = ALPHA_INCREASING

	p = index.Property()
	idx = index.Index(properties=p, interleaved=True)
	for i, key in enumerate(blocks):
		idx.insert(i, blocks[key].bounds)

	min_cost, min_stats = costfunc(list(blocks), blocks, idx, nets, mod2net, [l1,l2])

	cost_history = [min_cost]
	temp_cost = 1
	temp_stats = min_stats
	for i in tqdm(range(100), desc='annealer: ' + str(pos), position=pos, leave=False): # 1250
		"""
		ch = cost_history[:10]
		delta_cost = [ch[n]-ch[n-1] for n in range(1,len(ch))]
		convergence = True
		for d in delta_cost:
			if d > 10:
				convergence = False
		if convergence and i > 100:
			tqdm.write('converged...')
			break
		"""
		for ii in range(10): # 25
			tempBlocks, idx, temp_cost, temp_stats, updated_modules = transition(blocks,
																			   idx, board_dim,
																			   nets, mod2net,
																			   board_pins,
																			   i,
																			   min_cost, min_stats,
																			   costfunc,
																			   [l1,l2])
			delta = temp_cost - min_cost
			if (delta<0):
				blocks = tempBlocks
				min_cost = temp_cost
				min_stats = temp_stats
				cost_history.append(min_cost)
				deupdate(blocks, idx, updated_modules, 0)
			else:
				p = np.exp(-delta / T)
				if(np.random.random()<p):
					blocks = tempBlocks
					min_cost = temp_cost
					min_stats = temp_stats
					cost_history.append(min_cost)
					deupdate(blocks, idx, updated_modules, 0)
				else:
					cost_history.append(min_cost)
					deupdate(blocks, idx, updated_modules, 1)
		T,l1,l2,alpha,alpha_increasing = update_parameters(T,l1,l2,alpha,alpha_increasing)
	return blocks, min_cost, cost_history, min_stats, T, idx


def worker(args, output):
	output.put(annealing(*args))

def multistart(grid, 
			  connections, 
			  static_components, 
			  mod2net,
			  board_dim,
			  costfunc,
			  K=5,
			  seed=None):
	#num_cpus = min(4,mp.cpu_count())
	num_cpus = 4
	best_grid = grid
	best_cost = 100000000
	best_stats = []
	cost_history = []
	global_a = 0.5
	T_0 = INF
	board_dim = [[0,board_dim[0],board_dim[0],0],[0,0,board_dim[1],board_dim[1]]]
	for k in tqdm(range(K),desc='multi-start'):
		processes = []
		manager = mp.Manager()
		output = manager.Queue()
		for i in range(num_cpus):
			p = mp.Process(target=worker, args=((best_grid,connections,mod2net,static_components,board_dim,i+1,T_0,costfunc,seed),output))
			processes.append(p)
			p.start()

		for p in processes:
			p.join()
		results = [output.get() for p in processes]
		best_result = max(results,key=itemgetter(1)) # max result by cost
		T_0 = best_result[4]*1.25
		best_idx = best_result[5]
		if best_result[1] < best_cost:
			best_cost = best_result[1]
			best_grid = best_result[0]
			cost_history.extend(best_result[2])
			best_stats = best_result[3]
		else:
			cost_history.extend([best_cost]*1000)
	print('done')
	best_cost = cost(list(best_grid), best_grid, best_idx, connections, mod2net, [L1_eval,L2_eval])
	return best_grid, best_cost, cost_history, best_stats