import sys
sys.path.insert(1, '../src')

# Load our functions

from define_functions import *

# Import scipy.io for loading Matlab files

import scipy.io

for n in range(2, 11):

    mat = scipy.io.loadmat('../data/strengthened_sdr/data/TTRS__seed_1331__n_' + str(n) + '__N_15000_0.mat')

    # Make a first pass over instances to find how many were unsolved by Anders

    to_allocate = 0

    for i in range(0, mat["probs"].size):

        if mat["probs"][i, 0]['pksoc'][0, 0]['isExact'][0, 0][0, 0] == 0 \
            and mat["probs"][i, 0]['pksoc'][0, 0]['closed'][0, 0][0, 0] == 0:

            to_allocate = to_allocate + 1

    # Now go back over instances and solve them

    return_codes = [None] * to_allocate
    pvals        = -np.Inf * np.ones((to_allocate))
    dvals        = -np.Inf * np.ones((to_allocate))
    rel_gaps     = -np.Inf * np.ones((to_allocate))
    eval_ratios  = -np.Inf * np.ones((to_allocate))

    nxt = 0

    for i in range(0, mat["probs"].size):

        if mat["probs"][i, 0]['pksoc'][0, 0]['isExact'][0, 0][0, 0] == 0 \
            and mat["probs"][i, 0]['pksoc'][0, 0]['closed'][0, 0][0, 0] == 0:

            n = mat["probs"][i, 0]['pksoc'][0, 0]['data'][0, 0]['n'][0, 0][0, 0]
            H = mat["probs"][i, 0]['pksoc'][0, 0]['data'][0, 0]['H'][0, 0]
            g = mat["probs"][i, 0]['pksoc'][0, 0]['data'][0, 0]['g'][0, 0]
            r = mat["probs"][i, 0]['pksoc'][0, 0]['data'][0, 0]['r'][0, 0][0, 0]
            R = mat["probs"][i, 0]['pksoc'][0, 0]['data'][0, 0]['R'][0, 0][0, 0]
            a = mat["probs"][i, 0]['pksoc'][0, 0]['data'][0, 0]['a'][0, 0][0, 0]
            b = mat["probs"][i, 0]['pksoc'][0, 0]['data'][0, 0]['b'][0, 0]
            c = mat["probs"][i, 0]['pksoc'][0, 0]['data'][0, 0]['c'][0, 0]
            xhat = mat["probs"][i, 0]['pksoc'][0, 0]['data'][0, 0]['xhat'][0, 0]

            # Note: Vector b should be zero

            inst = {
                "n"    : n,
                "tipe" : 'sqrt',
                "Q"    : H,
                "c"    : g,
                "h1"   : 2*c,
                "g1"   : (-(c.T) @ c) + a**2,
                "x0"   : xhat
            }

            relaxation_options = {
                "Shor"       : True,
                "RLT"        : True,
                "SOCRLT"     : True,
                "Kron"       : True,
                "singleRLT0" : False
            }

            lift = lift_instance(inst)
            lift_std_form = convert_lift_to_standard_form(lift)
            sdp = build_sdp_standard_form(lift_std_form, relaxation_options)
            results = solve_sdp_standard_form(lift_std_form, sdp)

            return_codes[nxt] = results["return_code"]
            # if results["return_code"] == 'PrimalInfeasible':
            #     print(i)
            #     print(a)
            #     print(b.T @ xhat - a - npl.norm(xhat - c))
            #     print(inst["g1"] + inst["h1"].T @ xhat - npl.norm(xhat))
            pvals[nxt] = results["pval"]
            dvals[nxt] = results["dval"]
            rel_gaps[nxt] = results["rel_gap"]
            eval_ratios[nxt] = results["eval_ratio"]

            if rel_gaps[nxt] > tol_rel_gap or eval_ratios[nxt] < tol_eval_ratio:
                tmp = inst
                tmp["paper_pval"] = pvals[nxt]
                tmp["paper_dval"] = dvals[nxt]
                tmp["paper_relgap"] = rel_gaps[nxt]
                tmp["paper_eval_ratio"] = eval_ratios[nxt]
                myname = os.path.basename(os.path.abspath(__file__)).split('.')[0]
                myname = myname + '_n_' + str(n) + '_index_' + str(i)
                #  scipy.io.savemat('./code/unsolved_by_paper_instances/' + myname + '.mat', tmp)
            
            nxt = nxt + 1

    tmp = np.logical_and(eval_ratios > tol_eval_ratio, rel_gaps < tol_rel_gap)
    print('n = ' + str(n) + ': Solved ' + str(np.sum(tmp)) + \
            ' out of ' + str(to_allocate))
