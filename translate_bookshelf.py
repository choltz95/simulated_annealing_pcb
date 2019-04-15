import load_bookshelf
import json

def old2new():
	pass

def new2old():
	pass

# test cases
test_case_dir = './bookshelf_translation_testcases/'
old_test_case = test_case_dir + 'old/'
new_test_case = test_case_dir + 'new/'

circuit_name = 'apte'

old_pl_fname = old_test_case + circuit_name + '.pl'
old_nets_fname = old_test_case + circuit_name + '.nets'
old_blocks_fname = old_test_case + circuit_name + '.blocks'

new_pl_fname = new_test_case + circuit_name + '.pl'
new_nets_fname = new_test_case + circuit_name + '.nets'
new_nodes_fname = new_test_case + circuit_name + '.nodes'

components = load_bookshelf.read_blocks(old_blocks_fname)
placed_components, board_pins = load_bookshelf.read_pl(old_pl_fname)
nets, mod2net = load_bookshelf.read_nets(old_nets_fname, components, board_pins)

print('old circuit loaded')

#print('components',components)
#print('placed',placed_components)
#print('board_pins',board_pins)
#print('nets',nets)
#print('mod2net',mod2net)

load_bookshelf.write_newpl(test_case_dir+circuit_name+'.pl',components,board_pins)
load_bookshelf.write_nodes(test_case_dir+circuit_name+'.nodes',components,board_pins)
load_bookshelf.write_newnets(test_case_dir+circuit_name+'.nets',nets,components)

