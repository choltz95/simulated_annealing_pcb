# simulated_annealing_pcb
Adaptive simulated annealing for floor-planning arbitrary polygons

- minimize wirelength & overlap
- use r-tree for spatial indexing & fast intersection computation
- use Shapely for exact intersection
- about 300 iterations per second on my macbook pro
- computational bottleneck from cost computation, class attribute access overhead
- TODO: include code for lipo tuning, multi-start, follow the winners
