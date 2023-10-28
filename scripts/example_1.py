import sys
sys.path.insert(1, '../src')

# Import my functions

from define_functions import *

# Import OS, time, and progress functions

import os.path
import time
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
#  matplotlib.rcParams['font.serif'] = 'cm'

#

n = 2
seed = 622

inst = generate_martinez_instance(n, seed)

# Extract data

Q = inst["Q"]
c = inst["c"]
h1 = inst["h1"]
g1 = inst["g1"]

# Convert to format in paper

Q = Q
q = c
c1 = 0.5 * h1
rho1 = np.sqrt(g1 + 0.25 * np.dot(h1.T, h1))[0, 0]

# Round

dec = 2
Q = np.round(Q, decimals = dec)
q = np.round(q, decimals = dec)
c1 = np.round(c1, decimals = dec)
rho1 = np.round(rho1, decimals = dec)

# Manually change

Q[0, 1] = 0
Q[1, 0] = 0
q[1] = 0
c1[1] = -0.3
c1[0] = -0.3
Q[0, 0] = -0.6

print(Q)
print(q)
print(c1)
print(rho1)

# Data should be 
#  Q = [[-0.6000  0.0000]
#   [ 0.0000 -0.4400]]
#  q = [[-0.0300]
#   [ 0.0000]]
#  c1 = [[-0.3000]
#   [-0.3000]]
#  rho1 = 1.0

# Convert back to code format

Q = Q
c = q
h1 = 2 * c1
g1 = rho1**2 - np.dot(c1.T, c1)

# Copy back into inst

inst["Q"] = Q
inst["c"] = c
inst["h1"] = h1
inst["g1"] = g1

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

print(results["dval"])

relaxation_options = {
    "Shor"       : True,
    "RLT"        : False,
    "SOCRLT"     : False,
    "Kron"       : True,
    "singleRLT0" : False
}

inst_std_form = convert_inst_to_standard_form(inst)
sdp = build_sdp_standard_form(inst_std_form, relaxation_options)
results = solve_sdp_standard_form(inst_std_form, sdp)

#  print(results)
print(results["dval"])

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

print(results["dval"])
print(results["x"])

Y = np.arange(-1.0, 0.6, 0.01)
Z_Glob = np.Inf * Y
Z_Shor = np.Inf * Y
Z_Kron = np.Inf * Y
Z_Beta = np.Inf * Y

q = lambda x, y: \
    - 0.6 * x**2 \
    - 0.44 * y**2 \
    - 2 * 0.03 * x

for i in range(0, Y.size):

    val = Y[i]

    # Global (aka "Glob")

    X = np.arange(-1, 1, 0.0001)

    idx1 = (X**2 + val**2 <= 1.0)
    idx2 = ((X + 0.3)**2 + (val + 0.3)**2 <= 1**2)
    idx = idx1 & idx2
    X = X[idx]

    Z = q(X, val * np.ones(X.size))

    Z_Glob[i] = np.amin(Z)

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

    M = sdp['M']
    X = sdp['X']
    M_con = sdp['M_con']
    M_con_str = sdp['M_con_str']

    mystr = 'X_20=fixed linear equation'
    M_con.append(M.constraint(mystr, X.index([2, 0]), Domain.equalsTo(val)))
    M_con_str.append(mystr)

    results = solve_sdp_standard_form(inst_std_form, sdp)
    
    Z_Shor[i] = results['dval']

    # Kron

    relaxation_options = {
        "Shor"       : True,
        "RLT"        : False,
        "SOCRLT"     : False,
        "Kron"       : True,
        "singleRLT0" : False
    }

    inst_std_form = convert_inst_to_standard_form(inst)
    sdp = build_sdp_standard_form(inst_std_form, relaxation_options)

    M = sdp['M']
    X = sdp['X']
    M_con = sdp['M_con']
    M_con_str = sdp['M_con_str']

    mystr = 'X_20=fixed linear equation'
    M_con.append(M.constraint(mystr, X.index([2, 0]), Domain.equalsTo(val)))
    M_con_str.append(mystr)

    results = solve_sdp_standard_form(inst_std_form, sdp)
    
    Z_Kron[i] = results['dval']

    # Beta

    relaxation_options = {
        "Shor"       : True,
        "RLT"        : True,
        "SOCRLT"     : True,
        "Kron"       : False,
        "singleRLT0" : False
    }

    lift = lift_instance(inst)
    lift_std_form = convert_lift_to_standard_form(lift)
    sdp = build_sdp_standard_form(lift_std_form, relaxation_options)

    M = sdp['M']
    X = sdp['X']
    M_con = sdp['M_con']
    M_con_str = sdp['M_con_str']

    mystr = 'X_20=fixed linear equation'
    M_con.append(M.constraint(mystr, X.index([2, 0]), Domain.equalsTo(val)))
    M_con_str.append(mystr)

    results = solve_sdp_standard_form(lift_std_form, sdp)
    
    Z_Beta[i] = results['dval']

# plt.clf()
fig, axs = plt.subplots(2, 2, layout = 'constrained', sharex = True, sharey = True)

fig.supxlabel("$x_2$")
fig.supylabel("Minimum value with $x_2$ fixed")

axs[0, 1].plot(Y, Z_Shor, linewidth = 2)

axs[1, 0].plot(Y, Z_Kron, linewidth = 2)
axs[1, 1].plot(Y, Z_Beta, linewidth = 2)

axs[0, 0].title.set_text("Original nonconvex quadratic")
axs[0, 1].title.set_text("{\sc Shor}")
axs[1, 0].title.set_text("{\sc Kron}")
axs[1, 1].title.set_text("{\sc Beta}")

axs[0, 0].plot(Y, Z_Glob, color = 'k', linewidth = 2)
axs[1, 0].plot(Y, Z_Glob, color = 'k', linestyle = 'dotted')
axs[0, 1].plot(Y, Z_Glob, color = 'k', linestyle = 'dotted')
axs[1, 1].plot(Y, Z_Glob, color = 'k', linestyle = 'dotted')

#  plt.show()

fig.savefig('../results/example_1.png', dpi = 600)
