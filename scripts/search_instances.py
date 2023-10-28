import sys
sys.path.insert(1, '../src')

# Import my functions

from define_functions import *              # Generator functions

# Import OS, time, and progress functions

import os.path
import time
from tqdm import tqdm

# Define filename to store results of search

tmp = os.path.splitext(os.path.basename(__file__))[0]
fname = '../results/' + tmp + '.csv'

# Check if CSV file exists. If not, then create with correct column names

if os.path.isfile(fname) == False:
    mystr = 'type,n,m,seed,generator_timestamp,test_timestamp'
    f = open(fname, "w")
    f.write(mystr)
    f.close()

# Get timestamp for the generator file

tmp = time.localtime(os.path.getmtime('../src/define_functions.py'))
gstamp = time.strftime('%Y-%m-%d %H:%M:%S', tmp)

# Specify number of tests to run

num_tests = 2500000 # 2500000 is default for revision 2023-04-21

# Specify target number of instances unsolved by Shor in each grouping

target_num_unsolved_by_shor = 1000
#  print('Warning: Need to reset target_num_unsolved_by_shor = 1000')

# Specify the type of test to run

#  tipes = ['linear', 'martinez', 'maxnorm']
tipes = ['martinez', 'maxnorm']

# Loop over tipes

for tipe in tipes:

    # Specify dimension n

    if tipe == 'linear':
        ns = [2, 4, 6]
    elif tipe == 'martinez':
        ns = [2, 4, 6]
    elif tipe == 'maxnorm':
        ns = [2, 4]
    else:
        print('Incorrect tipe')
        exit()

    for n in ns:

        # Specify dimension m

        if tipe == 'linear':
            ms = [1]
        elif tipe == 'martinez':
            ms = [1]
        elif tipe == 'maxnorm' and n == 2:
            ms = [4, 8]
        elif tipe == 'maxnorm' and n == 4:
            ms = [8]
        else:
            print('Incorrect tipe')
            exit()

        for m in ms:

            # Read CSV file

            df = pd.read_csv(fname)

            # Filter CSV file to just tipe, and get largest seed so far
            # (plus 1). Or set to 0 if there are no records

            tmp = df[df['type'] == tipe]
            tmp = tmp[tmp['n'] == n]
            tmp = tmp[tmp['m'] == m]

            current_cnt = len(tmp.index)

            if current_cnt < target_num_unsolved_by_shor:

                if current_cnt > 0:
                    seed_start = tmp['seed'].max() + 1
                else:
                    seed_start = 0

                for i in tqdm(range(seed_start, seed_start + num_tests)):

                    #  if tipe == 'linear':
                    #      inst = generate_random_instance(n, 'lin', i)
                    if tipe == 'martinez':
                        inst = generate_martinez_instance(n, i)
                    elif tipe == 'maxnorm':
                        inst = generate_maxnorm_instance(n, m, i)
                    else:
                        print('Incorrect tipe')
                        exit()

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
                        solved_by_shor = 1
                    else:
                        solved_by_shor = 0
                        current_cnt = current_cnt + 1
                        tmp = time.localtime()
                        tmp = time.strftime('%Y-%m-%d %H:%M:%S', tmp)
                        df.loc[len(df.index)] = [tipe, n, m, i, gstamp, tmp]

                    if current_cnt >= target_num_unsolved_by_shor:
                        break

            # End if en(tmp.index) < target_num_unsolved_by_shor

            df.to_csv(fname, index = False)

            tmp = df[df['type'] == tipe]
            tmp = tmp[tmp['n'] == n]
            tmp = tmp[tmp['m'] == m]
            print('Num instances unsolved by Shor for (%s,%d,%d) = %d' % (tipe, n, m, len(tmp.index)))

        # End for m in ms

    # End for n in ns

# End for tipe in tipes
