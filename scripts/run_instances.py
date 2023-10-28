import sys
sys.path.insert(1, '../src')

from define_functions import *

# Import scipy.io for reading/writing Matlab files

import scipy.io

# Import OS, time, and progress functions

import os.path
import time
from tqdm import tqdm

# Import copy

import copy

# Define filename with instances to tests

fname = '../results/search_instances.csv'

# Check if CSV file exists. If not, then exit

if os.path.isfile(fname) == False:
    print('CSV file with instances does not exist')
    exit()

# Set number we would like to solve in each subcategory

target_num_unsolved_by_shor = 1000
#  print('Warning: Need to reset target_num_unsolved_by_shor = 1000')

# Read and filter data frame containing instances

df = pd.read_csv(fname)

#  Filter out linear instances (may be lingering from old version of code)

df = df[ (df.type != "linear") ]

# Filter out big instances (may be lingering from old version of code)

df = df[ ((df.type != "maxnorm") | (df.n != 4) | (df.m != 16)) ]

# Sort in such a way that we get first target_num_unsolved_by_shor
# instances using the seed column

# https://towardsdatascience.com/python-pandas-vs-r-dplyr-5b5081945ccb
df = (df.groupby(['type', 'n', 'm'], group_keys = False) \
    .apply( \
    lambda x: x.nsmallest( \
    target_num_unsolved_by_shor, 'seed' \
    )))

################################################################################

to_allocate = 3 * len(df.index) # Because three methods per instance

relax        = [None] * to_allocate
return_codes = [None] * to_allocate
dims1        = -np.Inf * np.ones((to_allocate))
dims2        = -np.Inf * np.ones((to_allocate))
tipes        = [None] * to_allocate
seeds        = -np.Inf * np.ones((to_allocate))
pvals        = -np.Inf * np.ones((to_allocate))
dvals        = -np.Inf * np.ones((to_allocate))
rel_gaps     = -np.Inf * np.ones((to_allocate))
eval_ratios  = -np.Inf * np.ones((to_allocate))
times        = -np.Inf * np.ones((to_allocate))

nxt = 0

for k in tqdm(range(0, len(df.index))):

    tipe = df['type'].values[k]
    n = df['n'].values[k]
    m = df['m'].values[k]
    seed = df['seed'].values[k]
    
    if tipe == 'linear':
        inst = generate_random_instance(n, 'lin', seed)
    elif tipe == 'martinez':
        inst = generate_martinez_instance(n, seed)
    elif tipe == 'maxnorm':
        inst = generate_maxnorm_instance(n, m, seed)
    else:
        print('Error: Incorrect tipe')
        exit(0)

    # Shor

    relaxation_options = {
        "Shor"       : True,
        "RLT"        : False,
        "SOCRLT"     : False,
        "Kron"       : False,
        "singleRLT0" : False
    }

    inst_std_form = convert_inst_to_standard_form(inst)
    sdp = build_sdp_standard_form(inst_std_form, relaxation_options)
    results = solve_sdp_standard_form(inst_std_form, sdp)

    rg = results["rel_gap"]
    er = results["eval_ratio"]

    if rg < tol_rel_gap and er > tol_eval_ratio:
        solved_by_shor = 1 # This should not occur because these instances are designed as such
    else:

        solved_by_shor = 0

        # Save Shor results

        relax       [nxt] = 'shor'
        dims1       [nxt] = n
        dims2       [nxt] = m
        tipes       [nxt] = tipe
        seeds       [nxt] = seed
        return_codes[nxt] = results["return_code"]
        pvals       [nxt] = results["pval"]
        dvals       [nxt] = results["dval"]
        rel_gaps    [nxt] = results["rel_gap"]
        eval_ratios [nxt] = results["eval_ratio"] 
        times       [nxt] = results["time"]

        inst["results_shor"] = results

        nxt = nxt + 1

        # Solve and save the "direct", or "original", Kron
        # method based on doing Kronecker products of pairs of
        # the form ||x - c|| <= rho

        #  sdp_no_beta = build_sdp_no_beta(lift_no_beta, 'kron_orig')
        #  results = solve_sdp_no_beta(lift_no_beta, sdp_no_beta)
        #
        #  relax       [nxt] = 'kron_orig'
        #  dims1       [nxt] = n
        #  dims2       [nxt] = m
        #  tipes       [nxt] = tipe
        #  seeds       [nxt] = seed
        #  return_codes[nxt] = results["return_code"]
        #  pvals       [nxt] = results["pval"]
        #  dvals       [nxt] = results["dval"]
        #  rel_gaps    [nxt] = results["rel_gap"]
        #  eval_ratios [nxt] = results["eval_ratio"]
        #  times       [nxt] = results["time"]
        #
        #  inst["results_kron_orig"] = results
        #
        #  nxt = nxt + 1

        # Solve and save Kron

        # We do not solve this for large instances because it takes too
        # long and will not be shown in paper

        if tipe != "maxnorm" or (n <= 4 and m <= 9): # Note: Hard-coded bounds on size for max-norm

            relaxation_options = {
                "Shor"       : True,
                "RLT"        : True,
                "SOCRLT"     : True,
                "Kron"       : True,
                "singleRLT0" : False
            }

            sdp = build_sdp_standard_form(inst_std_form, relaxation_options)
            results = solve_sdp_standard_form(inst_std_form, sdp)

            relax       [nxt] = 'kron'
            dims1       [nxt] = n
            dims2       [nxt] = m
            tipes       [nxt] = tipe
            seeds       [nxt] = seed
            return_codes[nxt] = results["return_code"]
            pvals       [nxt] = results["pval"]
            dvals       [nxt] = results["dval"]
            rel_gaps    [nxt] = results["rel_gap"]
            eval_ratios [nxt] = results["eval_ratio"]
            times       [nxt] = results["time"] # Watch this warning: "ComplexWarning: Casting complex values to real discards the imaginary part"

            kron_dval = results["dval"] # For exporting to Matlab below

            inst["results_kron"] = results

        else:

            relax       [nxt] = 'kron'
            dims1       [nxt] = n
            dims2       [nxt] = m
            tipes       [nxt] = tipe
            seeds       [nxt] = seed
            return_codes[nxt] = 'NotSolved'
            pvals       [nxt] = np.inf
            dvals       [nxt] = -np.inf
            rel_gaps    [nxt] = np.inf
            eval_ratios [nxt] = -np.inf
            times       [nxt] = -np.inf
            kron_dval = -np.inf # For exporting to Matlab below

        nxt = nxt + 1

        # Solve and save beta model of paper

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

        relax       [nxt] = 'beta'
        dims1       [nxt] = n
        dims2       [nxt] = m
        tipes       [nxt] = tipe
        seeds       [nxt] = seed
        return_codes[nxt] = results["return_code"]
        pvals       [nxt] = results["pval"]
        dvals       [nxt] = results["dval"]
        rel_gaps    [nxt] = results["rel_gap"]
        eval_ratios [nxt] = results["eval_ratio"] 
        times       [nxt] = results["time"] # Watch this warning: "ComplexWarning: Casting complex values to real discards the imaginary part"

        inst["results_beta"] = results

        # Save instance if not even solved by beta

        if rel_gaps[nxt] > tol_rel_gap or eval_ratios[nxt] < tol_eval_ratio:
            tmp = copy.deepcopy(inst)
            tmp["kron_dval"]       = kron_dval
            tmp["beta_pval"]       = pvals[nxt]
            tmp["beta_dval"]       = dvals[nxt]
            tmp["beta_relgap"]     = rel_gaps[nxt]
            tmp["beta_eval_ratio"] = eval_ratios[nxt]
            myname = os.path.basename(os.path.abspath(__file__)).split('.')[0]
            myname = myname + '_' + str(tipe) + '_n_' + str(n) + '_m_' + str(m) + '_seed_' + str(seed)
            scipy.io.savemat('../results/instances_unsolved_by_beta/' + myname + '.mat', tmp)

        # End if

        nxt = nxt + 1

        # Code to save instance, even if solved by Beta (i.e., by paper's relaxation)

        #  tmp2 = copy.deepcopy(inst)
        #  myname = os.path.basename(os.path.abspath(__file__)).split('.')[0]
        #  myname = myname + '_type_' + str(tipe) + '_n_' + str(n) + '_m_' + str(m) + '_seed_' + str(seed)
        #  scipy.io.savemat('./mat/instances/' + myname + '.mat', tmp2)

    # End if solved_by_shor == 0

# End for loop over rows of data frame (index k)

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
