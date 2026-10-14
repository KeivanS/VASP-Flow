# Niggli-reduced cell of a POSCAR (numpy only): python3 reduce_cell.py IN [OUT]
# (OUT default: IN, original kept as IN.input).  The agents and ht_runs.py already do this
# when LOBSTER runs; this is for runs staged before v2.6.  Code: modules/cell_reduce.py
import os, runpy, sys
sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modules', 'cell_reduce.py')
runpy.run_path(sys.argv[0], run_name='__main__')
