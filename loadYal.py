import yaml
import json


def space_level(astr):
    """Count number of leading tabs in a string
    """
    return len(astr)- len(astr.lstrip(' '))

def load_yal(fname):
	iolist = -1
	with open(fname) as f:       
		d = {}
		fstr = f.read().replace('\n', '').split(';')
		for line in fstr:
			sl = space_level(line)
			if sl == 0:
				module = line.replace('MODULE ', '')
				d[module] = {}
			elif sl == 1:
				if 'TYPE' in line:
					d[module]['type'] = line.split()[1]
				elif 'DIMENSIONS' in line:
					d[module]['dimensions'] = line.split()[1:]
				elif ' IOLIST' in line:
					iolist = 1
					d[module]['iolist'] = []
				elif ' NETWORK' in line:
					iolist = 0
					d[module]['network'] = []
			elif sl == 2:
				if iolist == 1:
					d[module]['iolist'].append(line.strip().replace("    "," "))
				elif iolist == 0:
					d[module]['network'].append(line.strip().replace("    "," "))

	return d