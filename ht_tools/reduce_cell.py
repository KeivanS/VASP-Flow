# Shortest-vector cell reduction of a POSCAR (no pymatgen): python3 reduce_cell.py IN OUT
import sys, itertools
import numpy as np
L = open(sys.argv[1]).read().splitlines()
s = float(L[1]); A = np.array([[float(x) for x in L[i].split()[:3]] for i in range(2, 5)]) * s
el, cnt = L[5].split(), [int(x) for x in L[6].split()]
n = sum(cnt); i0 = 8 if L[7].strip()[0] in 'Ss' else 7
cart = L[i0].strip()[0] in 'CcKk'
X = np.array([[float(x) for x in L[i0 + 1 + k].split()[:3]] for k in range(n)])
F = X @ np.linalg.inv(A) if cart else X
R = range(-4, 5)
cands = sorted((np.linalg.norm(np.array(c) @ A), c) for c in itertools.product(R, R, R) if any(c))
M = None
for _, c1 in cands:                      # greedy: shortest 3 independent with |det| = 1
    for _, c2 in cands:
        if abs(np.linalg.norm(np.cross(c1, c2))) < 1e-9: continue
        for _, c3 in cands:
            if round(abs(np.linalg.det(np.array([c1, c2, c3])))) == 1:
                M = np.array([c1, c2, c3]); break
        if M is not None: break
    if M is not None: break
if np.linalg.det(M) < 0: M[2] = -M[2]
B = M @ A
Fn = (F @ np.linalg.inv(M)) % 1.0
ab = np.linalg.norm(B, axis=1); ang = lambda i, j: np.degrees(np.arccos(B[i] @ B[j] / ab[i] / ab[j]))
print('original |a|', np.round(np.linalg.norm(A, axis=1), 3), ' volume %.3f' % abs(np.linalg.det(A)))
print('reduced  |a|', np.round(ab, 3), ' angles %.1f %.1f %.1f' % (ang(1, 2), ang(0, 2), ang(0, 1)),
      ' volume %.3f' % abs(np.linalg.det(B)))
with open(sys.argv[2], 'w') as f:
    f.write(L[0] + ' (reduced cell)\n1.0\n' + ''.join('  %.10f %.10f %.10f\n' % tuple(v) for v in B))
    f.write(L[5] + '\n' + L[6] + '\nDirect\n' + ''.join('  %.10f %.10f %.10f\n' % tuple(p) for p in Fn))
