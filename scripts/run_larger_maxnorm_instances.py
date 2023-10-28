import sys
sys.path.insert(1, '../src')

# Import my functions

from define_functions import *

# Import OS, time, and progress functions

import os.path
import time
from tqdm import tqdm

# Define filename to store results of search

tmp = os.path.splitext(os.path.basename(__file__))[0]
fname = '../results/' + tmp + '.csv'

# Check if CSV file exists. If not, then create with correct column names

if os.path.isfile(fname) == False:
    mystr = 'type,n,m,seed,generator_timestamp,test_timestamp,relaxation,time'
    f = open(fname, "w")
    f.write(mystr)
    f.close()

# Get timestamp for the generator file

tmp = time.localtime(os.path.getmtime('../src/define_functions.py'))
gstamp = time.strftime('%Y-%m-%d %H:%M:%S', tmp)

# Specify number of tests to run

num_tests = 100 

# Specify the type of test to run

tipe = 'maxnorm'

# Specify dimension n

ns = [2, 4, 8, 16, 32, 64]

for n in ns:

    # Specify dimension m

    ms = [2, 4, 8, 16, 32, 64]

    for m in ms:

        # Read CSV file

        df = pd.read_csv(fname)

        seed_start = 0

        for i in tqdm(range(seed_start, seed_start + num_tests)):

            inst = generate_maxnorm_instance(n, m, i)

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

            tmp = time.localtime()
            tmp = time.strftime('%Y-%m-%d %H:%M:%S', tmp)
            df.loc[len(df.index)] = [tipe, n, m, i, gstamp, tmp, 'beta', results["time"]]

        df.to_csv(fname, index = False)

    # End for m in ms

# End for n in ns
