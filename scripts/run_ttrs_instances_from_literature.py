import sys
sys.path.insert(1, '../src')

from define_functions import *

# Import scipy.io for reading/writing Matlab files

import scipy.io

# Import OS, glob, time, and progress functions

import os.path
import glob
import time
from tqdm import tqdm

# Import copy

import copy

relaxation_options = {
    "Shor"       : True,
    "RLT"        : True,
    "SOCRLT"     : True,
    "Kron"       : True,
    "singleRLT0" : True
}

dir_list = glob.glob('../data/soctrust/Section_5_3_unsolved_instances_saved/instance_*.mat')

to_allocate = len(dir_list) # Because three methods per instance

relax        = [None] * to_allocate
return_codes = [None] * to_allocate
dims1        = -np.Inf * np.ones((to_allocate))
dims2        = -np.Inf * np.ones((to_allocate))
tipes        = [None] * to_allocate
seeds        = [None] * to_allocate
pvals        = -np.Inf * np.ones((to_allocate))
dvals        = -np.Inf * np.ones((to_allocate))
rel_gaps     = -np.Inf * np.ones((to_allocate))
eval_ratios  = -np.Inf * np.ones((to_allocate))
times        = -np.Inf * np.ones((to_allocate))

nxt = 0

for fname in dir_list:

    mat = scipy.io.loadmat(fname)

    # min   x'Qx + c'x  s.t.    ||x|| <= r1     ||sqrt(H)(x - h)|| <= r2

    # These instances have h=0 and r1=r2=n

    Q = mat["Q"]
    c = mat["c"]
    H = mat["H"]
    h = mat["h"]
    r1 = mat["r1"]
    r2 = mat["r2"]

    n = np.shape(Q)[0]

    if np.abs(n - r1) > tol_general or np.abs(n - r2) > tol_general:
        print("Unexpected value for radius")
        exit(0)

    if npl.norm(h) > tol_general:
        print("Unexpected value for center of second ellipsoid")
        exit(0)

    if npl.norm(np.diagflat(np.diag(H)) - H) > tol_general:
        print("Unexpected value for Hessian of second ellipsoid")
        exit(0)

    # Scale so that radii become 1

    # min   x'Qx + c'x  s.t.    ||x|| <= r1     ||sqrt(H)x|| <= r1

    # x gets replaced by r1 y

    # min   (r1 y)'Q(r1 y) + c'(r1 y)   s.t.    ||r1 y|| <= r1  ||sqrt(H) r1 y|| <= r1

    # min   y' * (r1^2 Q) * y + (r1 c)'y    s.t.    ||y|| <= 1  ||sqrt(H) y|| <= 1

    # In particular, optimal objective value has not changed

    Q = Q * r1**2
    c = c * r1

    tmp1 = np.hstack(([[0]], 0.5*c.T))
    tmp2 = np.hstack((0.5*c, Q))
    Q = np.vstack((tmp1, tmp2))

    Hsqrt = np.diagflat(np.sqrt(np.diag(H)))

    n = n + 1

    A1 = np.eye(n)
    A2 = np.hstack(([[1]], np.zeros((1, n-1))))
    A3 = np.hstack((np.zeros((n-1, 1)), Hsqrt))

    A = np.vstack((A1, A2, A3))

    K_dims = [n, n]
    K_types = ['soc', 'soc']

    x0 = np.vstack(([[1]], np.zeros((n-1, 1))))

    inst_std_form = {
        "n"                 : n,
        "Q"                 : Q,
        "A"                 : A,
        "K_dims"            : K_dims,
        "K_types"           : K_types,
        "x0"                : x0,
        "first_entry_1"     : True,
        "Shor_implies_feas" : True,
        "Shor_bounded"      : True
    }

    lift = lift_concentric_ttrs_instance(inst_std_form)
    sdp = build_sdp_standard_form(lift, relaxation_options)
    results = solve_sdp_standard_form(lift, sdp)
    
    rg = results["rel_gap"]
    er = results["eval_ratio"]
    if rg < tol_rel_gap and er > tol_eval_ratio:
        print(fname + "\t\tSolved!")
    else:
        print(fname + "\t\t:-O")

    relax       [nxt] = 'beta'
    dims1       [nxt] = n - 1
    dims2       [nxt] = 2
    tipes       [nxt] = 'ttrs'
    seeds       [nxt] = fname
    return_codes[nxt] = results["return_code"]
    pvals       [nxt] = results["pval"]
    dvals       [nxt] = results["dval"]
    rel_gaps    [nxt] = results["rel_gap"]
    eval_ratios [nxt] = results["eval_ratio"] 
    times       [nxt] = results["time"] 

    nxt = nxt + 1

# End for

df = {
    'type' : tipes,
    'n' : dims1,
    'extra_balls' : dims2,
    'seed' : seeds,
    'relaxation' : relax,
    'mosek_code' : return_codes,
    'pval' : pvals,
    'dval' : dvals,
    'rel_gap' : rel_gaps,
    'eval_ratio' : eval_ratios,
    'time' : times
}

df = pd.DataFrame(df)
tmp = os.path.splitext(os.path.basename(__file__))[0]
df.to_csv("../results/" + tmp + ".csv", index = False)
