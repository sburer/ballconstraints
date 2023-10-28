###############################################################################

# Import packages

import os
import sys
import math
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import pandas as pd
import mosek
from mosek.fusion import *
import gurobipy as gp
from gurobipy import GRB

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "% .4f" % x))

# Import copy

import copy

###############################################################################

# Set constants

from define_constants import *

###############################################################################

# Define function to solve TRS SDP

def build_and_solve_trs_sdp(n, Qc, r1):

    # Setup basic model

    M = Model('SDP model for TRS')

    # Define Y = [Y00, x'; x, X] as usual
        
    Y    = M.variable('Y', Domain.inPSDCone(n + 1))
    Y00  = Y.slice([0, 0], [1, 1]) # Could also be Y.index([0, 0])
    x    = Y.slice([1, 0], [n + 1, 1])
    X    = Y.slice([1, 1], [n + 1, n + 1])
    
    # Add standard constraints

    con_Y00 = M.constraint('Y00', Y00, Domain.equalsTo(1.0))
    con_trX = M.constraint('trX', Expr.sum(X.diag()), Domain.lessThan(r1**2)) # Note r1**2

    # Setup objective

    M.objective(ObjectiveSense.Minimize, Expr.dot(Qc, Y))

    # Setup options. Do I want to add tighter tolerances?

    M.setLogHandler(sys.stdout)
    M.setSolverParam("log", 0)
    M.setSolverParam("intpntCoTolPfeas", tol_mosek)
    M.setSolverParam("intpntCoTolDfeas", tol_mosek)
    M.setSolverParam("intpntCoTolRelGap", tol_mosek)

    # Solve

    M.solve()
    
    # Get return code. If we don't get 'PrimalAndDualFeasible', we balk. Or
    # if norm of x is too far from r1

    return_code = M.getProblemStatus(SolutionType.Default).name

    # Get solution values

    YY = np.reshape(Y.level(), (n + 1, n + 1))
    yy = YY[:, [0]]
    xx = yy[1:(n + 1), :]

    if(return_code == 'PrimalAndDualFeasible' and np.abs(npl.norm(xx) - r1) < tol_general):

        success = 1
        
        # Get primal and dual values

        pval = yy.T @ Qc @ yy # Should I evaluate feasibility of yy (and xx)?
        dval = M.dualObjValue()

        # Calculate relative gap as usual. Except that if pval and dval
        # are really close, we purposely do not allow negative values

        if np.abs(pval - dval) < tol_general:
            rel_gap = abs(pval - dval) / np.maximum(1.0, np.abs(pval + dval) / 2.0);
        else:
            rel_gap = (pval - dval) / np.maximum(1.0, np.abs(pval + dval) / 2.0);

        # Calculate eigenvalue ratio as usual. Could put this in its own
        # function
            
        evals, evecs = npl.eig(YY)
        evals = -np.sort(-evals)
        eval_ratio = evals[0] / np.abs(evals[1])
        
    else:
        
        success = 0

        xx = np.Inf * np.ones((n, 1))
        pval = np.Inf
        dval = -np.Inf
        rel_gap = np.Inf
        eval_ratio = -np.Inf
        
    M.dispose()

    return success, xx, pval, dval, rel_gap, eval_ratio

###############################################################################

# Define function to generate a Martinez instance. Procedure follows
# https://epubs.siam.org/doi/10.1137/110826862 as closely as possible,
# except as noted near the end

def generate_martinez_instance(n, seed = -99):

    # If user has entered seed (not equal to -99), then set the seed

    if seed != -99:
        npr.seed(seed)

    # Set the first radius

    r1 = n

    overall_success = 0

    while overall_success == 0:

        # Generate Q as a (sorted) diagonal matrix with entries uniform
        # in [-1,1]. Filter for Q with negative eigenvalue. I don't
        # think sorting really matters

        successQ = 0
        while successQ == 0:
            vals = np.sort(npr.uniform(-1, 1, (n, 1)))
            if np.amin(vals) < 0:
                successQ = 1
                Q = np.diagflat(vals)

        # Generate c with entries uniform in [-1,1]

        c = npr.uniform(-1, 1, (n, 1))

        # Put Q and c together in a block matrix

        Qc = np.block([[0, c.T], [c, Q]])

        # Solve TRS

        success, xx, pval, dval, rel_gap, eval_ratio = build_and_solve_trs_sdp(n, Qc, r1)

        # If Martinez subroutine came back as success, then proceed

        if success == 1:

            # Get complete QR decomposition of xx

            ret_q, ret_r = npl.qr(xx, 'complete')
            ret_q = ret_q.T

            # Alter Q and c such that xx = r1 * e1 is the TRS optimal solution

            Q = ret_q.T @ Q @ ret_q
            c = ret_q.T @ c

            # Rebuild Qc

            Qc = np.block([[0, c.T], [c, Q]])

            # Here is where we start to deviate from the
            # https://epubs.siam.org/doi/10.1137/110826862 version of
            # Martinez. We need our new constraint to have the form ||x
            # - h|| <= r2 = r1, while cutting off xx = r1 * e1. So we
            # simply make ||h|| <= r1 such that h1 < 0.

            h = npr.standard_normal((n, 1))
            h = npr.uniform() * r1 * h / npl.norm(h)
            if h[0, 0] > 0:
                h = -h

            # Finally, we need to make sure our radius for both balls is
            # 1.0. All we have to do to the data is scale h down by r1
            # and then set r1 = 1

            h = h / r1
            r1 = 1

            # I'm pretty sure x0 = 0 is feasible. Let's double check

            x0 = np.zeros((n, 1))
            if npl.norm(x0) > r1 or npl.norm(x0 - h) > r1:
                print('Something wrong. Double check')

            inst = {
                "n"    : n,
                "tipe" : 'sqrt',
                "Q"    : Q,
                "c"    : c,
                "h1"   : 2*h,
                "g1"   : r1**2 - h.T @ h,
                "x0"   : -np.Inf * np.ones((n, 1))
            }

            overall_success = 1

        else:

            inst = np.nan

    return inst

###############################################################################

def generate_maxnorm_instance(n, num_extra_balls, seed = -99):

    # If user has entered seed (not equal to -99), then set the seed

    if seed != -99:
        npr.seed(seed)

    # Generate a point z, which we'll calculate the max distance to. z
    # can be inside or outside the unit ball. In particular, norm of z
    # is between 0 and 2

    z = npr.standard_normal((n, 1))
    z = npr.uniform(0, 2) * z / npl.norm(z)

    # So we want to min -(x - z)'(x - z) = -x'x + 2z'x - z'z. So set Q =
    # -I and c = z

    Q = -np.eye(n)
    c = z

    # Generate centers h of new balls inside unit ball, i.e., norm
    # between 0 and 1

    h = npr.standard_normal((n, num_extra_balls)) # Store in n x num_extra_balls matrix
    for j in range(0, num_extra_balls):
        h[:, [j]] = npr.uniform() * h[:, [j]] / npl.norm(h[:, [j]])

    # Generate radius r1 of new balls between 0 and 1

    r1 = -np.Inf * np.ones((num_extra_balls, 1))
    for j in range(0, num_extra_balls):
        tmp = npl.norm(h[:, [j]])
        r1[j, 0] = tmp + npr.uniform(0, 1.5)

    h1 = 2*h
    g1 = r1
    for j in range(0, num_extra_balls):
        g1[j, 0] = r1[j, 0]**2 - h[:, [j]].T @ h[:, [j]]

    inst = {
        "n"    : n,
        "tipe" : 'sqrt',
        "Q"    : Q,
        "c"    : c,
        "h1"   : h1,
        "g1"   : g1,
        "x0"   : np.zeros((n, 1))
    }

    # Goal has been to build balls such that x0 = 0 is feasible. Let's
    # check

    if np.sum(g1 + h1.T @ np.zeros((n, 1)) >= 0) < num_extra_balls:
        print('Warning: Seems like 0 is infeasible!')

    return inst

###############################################################################

def lift_instance(inst):
    
    n    = inst["n"]
    tipe = inst["tipe"]
    Q    = inst["Q"]    
    c    = inst["c"]
    g1   = inst["g1"]
    h1   = inst["h1"]
    x0   = inst["x0"]
    
    z = np.zeros((n, 1))
    
    l0 = np.block([[1], [z], [-1]])
    if tipe == 'lin':
        l1 = np.block([[g1], [h1], [-1]])
    else:
        # Next line supresses beta b/c we will eventually represent the
        # constraint as beta^2 <= l1'*w. Is there a more elegant way to
        # handle?
        m = g1.shape[0]
        l1 = np.block([[g1.T], [h1], [np.zeros((1, m))]])
        
    P = np.block([[0, z.T, 1], [z, np.eye(n), z]])
    L = np.block([[1, z.T], [z, -np.eye(n)]])
    
    Qc = np.block([[0, c.T, 0], [c, Q, z], [0, z.T, 0]])
    
    w0 = np.block([[1], [x0], [npl.norm(x0)]])
    
    lift = {
        "np2"  : n + 2,
        "tipe" : tipe,
        "Qc"   : Qc,
        "l0"   : l0,
        "l1"   : l1,
        "P"    : P,
        "L"    : L,
        "w0"   : w0
    }

    return lift

###############################################################################

def build_sdp_standard_form(inst_std_form, relaxation_options):

    n                 = inst_std_form["n"]
    Q                 = inst_std_form["Q"]
    A                 = inst_std_form["A"]    
    K_dims            = inst_std_form["K_dims"]
    K_types           = inst_std_form["K_types"]
    x0                = inst_std_form["x0"]
    first_entry_1     = inst_std_form["first_entry_1"]
    Shor_implies_feas = inst_std_form["Shor_implies_feas"]

    # Define constants
    
    In = np.eye(n)
    Jn = -In
    Jn[0, 0] = 1

    ###########################################################################
    # Prepare handy-dandy shortcuts related to A matrix (called B and C)
    ###########################################################################

    B = {}
    C = {}
    D = {}
    J = {}
    curr_pos_k = 0

    for k in range(0, len(K_dims)):

        B[k] = A[ curr_pos_k : (curr_pos_k + K_dims[k]) , 0:n]

        if K_types[k] == 'soc':

            Bk_dim = np.shape(B[k])[0]

            for j in range(0, n):

                # Build constant matrix C[k, j], which represents the
                # portion contributed by x[j] to the SDP representation
                # of Bk * x in SOC

                # Extract j-th column of Bk

                Bkj = B[k][:, j]

                # Build arrow matrix based on Bkj

                tmp = Bkj[0] * np.eye((Bk_dim))
                for ind in range(1, Bk_dim):
                    tmp[ind, 0] = Bkj[ind]
                    tmp[0, ind] = Bkj[ind]
                C[k, j] = tmp

            J[k] = -np.eye((Bk_dim))
            J[k][0, 0] = 1
            # Maybe make J[k] sparse?

        if K_types[k] == 'rsoc':

            Bk_dim = np.shape(B[k])[0]

            for j in range(0, n):

                # Build constant matrix C[k, j], which represents the
                # portion contributed by x[j] to the SDP representation
                # of Bk * x in RSOC

                # This assumes we are working with RSOC, which has the
                # form [1/2, g + h'x, x] with the PSD form [ 1, x'; x,
                # (g + h'x)*I ], i.e., a size p RSOC becomes a size p-1
                # PSD matrix

                # Extract j-th column of Bk

                Bkj = B[k][:, j]

                # Build arrow matrix based on Bkj. Is "arrow" the correct word?

                tmp = Bkj[1] * np.eye((Bk_dim - 1))
                tmp[0, 0] = 2 * Bkj[0]
                for ind in range(2, Bk_dim):
                    tmp[ind - 1, 0] = Bkj[ind]
                    tmp[0, ind - 1] = Bkj[ind]
                C[k, j] = tmp

            J[k] = -np.eye((Bk_dim))
            J[k][0, 0] = 0
            J[k][1, 0] = 1
            J[k][0, 1] = 1
            J[k][1, 1] = 0

        curr_pos_k = curr_pos_k + K_dims[k]

    ###########################################################################
    # Build model
    ###########################################################################

    M = Model('Generic name')
    M_con = []
    M_con_str = []

    # Setup main variable

    X = M.variable('X', Domain.inPSDCone(n))

    # Setup derived variable

    x = X.slice([0, 0], [n, 1])

    # Set top-left entry to 1

    mystr = 'X_00=1 linear equation'
    M_con.append(M.constraint(mystr, X.index([0, 0]), Domain.equalsTo(1.0)))
    M_con_str.append(mystr)

    # Go through all cones to add constraints common to Shor and Kron.
    # Just the quadratics defining the SOC and RSOC constraints

    for k in range(0, len(K_dims)):

        if K_types[k] == 'soc' or K_types[k] == 'rsoc':

            mystr = K_types[k].upper() + ' linearized quad ineq [' + str(k) + ']'
            myexpr = Expr.dot(B[k].T @ J[k] @ B[k], X)
            M_con.append(M.constraint(mystr, myexpr, Domain.greaterThan(0.0)))
            M_con_str.append(mystr)

    # If Shor doesn't necessarily give a feasible first column, enforce
    # feasibility on the first column

    if Shor_implies_feas == False and relaxation_options["Shor"] == True and relaxation_options["Kron"] == False:

        for k in range(0, len(K_dims)):

            mystr = 'First col ' + K_types[k] + ' [' + str(k) + ']'
            myexpr = Expr.flatten(Expr.mul(B[k], x))

            if K_types[k] == 'nonneg':

                M_con.append(M.constraint(mystr, myexpr, Domain.greaterThan(0)))
                M_con_str.append(mystr)

            if K_types[k] == 'soc':

                M_con.append(M.constraint(mystr, myexpr, Domain.inQCone( np.shape(B[k])[0] )))
                M_con_str.append(mystr)

            elif K_types[k] == 'rsoc':

                M_con.append(M.constraint(mystr, myexpr, Domain.inRotatedQCone( np.shape(B[k])[0] )))
                M_con_str.append(mystr)

    # Go through pairs of cones...

    for k in range(0, len(K_dims)):

        Bk = B[k]
        Bk_dim = np.shape(Bk)[0]

        for l in range(0, k + 1):

            Bl = B[l]
            Bl_dim = np.shape(Bl)[0]

            str_kl = ' [cone ' + str(k) + ', cone ' + str(l) + '] '

            # RLT

            if relaxation_options["RLT"] == True and K_types[k] == 'nonneg' and K_types[l] == 'nonneg' and k == l:

                for i in range(0, Bk_dim):
                    for j in range(0, i):
                        str_ij = '[lin ineq ' + str(i) + ', lin ineq ' + str(j) + '] '
                        mystr = 'RLT' + str_kl + str_ij
                        myexpr = Expr.dot(Bk[i,:], Expr.mul(X, Bk[j,:].T))
                        # This is a bit hacky for concentric TTRS when base dim is 2  --Sam 2023-05-24
                        if relaxation_options["singleRLT0"] == False:
                            M_con.append(M.constraint(mystr, myexpr, Domain.greaterThan(0)))
                        else:
                            M_con.append(M.constraint(mystr, myexpr, Domain.equalsTo(0)))
                        M_con_str.append(mystr)

            if relaxation_options["RLT"] == True and K_types[k] == 'nonneg' and K_types[l] == 'nonneg' and k != l:

                mystr = 'RLT' + str_kl
                myexpr = Expr.mul(Bl, Expr.mul(X, Bk.T))
                M_con.append(M.constraint(mystr, myexpr, Domain.greaterThan(0)))
                M_con_str.append(mystr)

            # SOCRLT

            if relaxation_options["SOCRLT"] == True and K_types[k] == 'nonneg' and K_types[l] == 'soc':

                for j in range(0, K_dims[k]):
                    mystr = 'SOCRLT [' + str(k) + ', ' + str(l) + ', ', + str(j) + ']'
                    myexpr = Expr.mul(Bl, Expr.mul(X, Bk[j, :].T))
                    M_con.append(M.constraint(mystr, myexpr, Domain.inQCone(Bl_dim)))
                    M_con_str.append(mystr)

            if relaxation_options["SOCRLT"] == True and K_types[k] == 'soc' and K_types[l] == 'nonneg':

                for j in range(0, K_dims[l]):
                    myexpr = Expr.mul(Bk, Expr.mul(X, Bl[j, :].T))
                    mystr = 'SOCRLT [' + str(k) + ', ' + str(l) + ', ' + str(j) + ']'
                    M_con.append(M.constraint(mystr, myexpr, Domain.inQCone(Bk_dim)))
                    M_con_str.append(mystr)

            # RSOCRLT

            if relaxation_options["SOCRLT"] == True and K_types[k] == 'nonneg' and K_types[l] == 'rsoc':

                for j in range(0, K_dims[k]):
                    str_j = '[' + str(j) + '] '
                    mystr = 'RSOCRLT' + str_kl + str_j
                    myexpr = Expr.mul(Bl, Expr.mul(X, Bk[j, :].T))
                    M_con.append(M.constraint(mystr, myexpr, Domain.inRotatedQCone(Bl_dim)))
                    M_con_str.append(mystr)

            if relaxation_options["SOCRLT"] == True and K_types[k] == 'rsoc' and K_types[l] == 'nonneg':

                for j in range(0, K_dims[l]):
                    str_j = '[lin ineq ' + str(j) + '] '
                    mystr = 'RSOCRLT' + str_kl + str_j
                    myexpr = Expr.mul(Bk, Expr.mul(X, Bl[j, :].T))
                    M_con.append(M.constraint(mystr, myexpr, Domain.inRotatedQCone(Bk_dim)))
                    M_con_str.append(mystr)

            # Kron

            if relaxation_options["Kron"] and \
                    (K_types[k] == 'soc' or K_types[k] == 'rsoc') and \
                    (K_types[l] == 'soc' or K_types[l] == 'rsoc') and k != l:

                Ck_dim = np.shape(C[k, 0])[0]
                Cl_dim = np.shape(C[l, 0])[0]

                mymat = np.zeros((Ck_dim * Cl_dim, Ck_dim * Cl_dim))

                for i in range(0, n):
                    for j in range(0, n):
                        myexpr = np.kron(C[k, i], C[l, j])
                        myexpr = Expr.mul(X.index([i, j]), myexpr)
                        mymat = Expr.add(mymat, myexpr)

                #  mystr = 'Kron' + str_kl
                #  M_con.append(M.constraint(mystr, mymat, Domain.inPSDCone( Ck_dim * Cl_dim )))
                #  M_con_str.append(mystr)

                myleft = np.zeros([ (Ck_dim * Cl_dim)**2 , 0])
                for i in range(0, n):
                    for j in range(0, n):
                        mydata = np.kron(C[k, i], C[l, j])
                        mydata = np.reshape(mydata, ((Ck_dim * Cl_dim)**2, 1))
                        myleft = np.hstack((myleft, mydata))

                myexpr = Expr.reshape(X, n**2, 1)
                myexpr = Expr.mul(myleft, myexpr)
                myexpr = Expr.reshape(myexpr, Ck_dim * Cl_dim, Ck_dim * Cl_dim)

                mystr = 'Kron' + str_kl
                M_con.append(M.constraint(mystr, myexpr, Domain.inPSDCone( Ck_dim * Cl_dim )))
                M_con_str.append(mystr)

            # End all the cases

        # End loop over l

    # End loop over k

    M.objective(ObjectiveSense.Minimize, Expr.dot(Q, X))
    
    M.setLogHandler(sys.stdout)
    M.setSolverParam("log", 0)
    M.setSolverParam("intpntCoTolPfeas", tol_mosek)
    M.setSolverParam("intpntCoTolDfeas", tol_mosek)
    M.setSolverParam("intpntCoTolRelGap", tol_mosek)
    
    sdp = {
        "M" : M,
        "X" : X,
        "M_con" : M_con,
        "M_con_str" : M_con_str
    }

    return sdp

###############################################################################

def solve_sdp_standard_form(inst_std_form, sdp):

    n = inst_std_form["n"]
    Q = inst_std_form["Q"]
    A = inst_std_form["A"]

    M = sdp["M"]
    X = sdp["X"]
    
    M.solve()
    
    return_code = M.getProblemStatus(SolutionType.Default).name
    tm = M.getSolverDoubleInfo("optimizerTime")

    if(return_code == 'PrimalAndDualFeasible'):

        XX = np.reshape(X.level(), (n, n))
        XX = 0.5*(XX + XX.T)
        xx = XX[:, [0]]

        pval = xx.T @ Q @ xx
        pval = pval[0, 0]

        dval = M.dualObjValue()
        rel_gap = (pval - dval) / np.maximum(1.0, np.abs(pval + dval) / 2.0)

        if rel_gap < 0 and np.abs(rel_gap) < tol_general:
            rel_gap = -rel_gap

        evals, evecs = npl.eigh(XX)
        evals = -np.sort(-evals)
        eval_ratio = evals[0] / np.abs(evals[1])

        XX_dual = np.reshape(X.dual(), (n, n))
        XX_dual = 0.5 * (XX_dual + XX_dual.T)

    else:

        pval = np.Inf
        dval = -np.Inf
        rel_gap = np.Inf
        eval_ratio = -np.Inf
        XX = -np.Inf
        xx = -np.Inf

    #  M.writeTask('dump.ptf')
    M.dispose()
    #  print("Not disposing of M")
        
    results = {
        "return_code" : return_code,
        "x" : xx,
        "X" : XX,
        "X_dual" : XX_dual,
        "pval" : pval,
        "dval" : dval,
        "rel_gap" : rel_gap,
        "eval_ratio" : eval_ratio,
        "time" : tm
    }
    
    return results

###############################################################################

def convert_inst_to_standard_form(inst):

    tmp1 = np.hstack(([[0]], inst["c"].T))
    tmp2 = np.hstack((inst["c"], inst["Q"]))
    Q = np.vstack((tmp1, tmp2))

    n = inst["n"] + 1

    if inst["tipe"] == 'lin':

        tmp1 = np.eye(n)
        tmp2 = np.hstack((inst["g1"], inst["h1"].T))
        tmp3 = np.hstack((np.zeros((n-1, 1)), np.eye(n-1)))
        A = np.vstack((tmp1, tmp2, tmp3))

        K_dims = [n, n]

        K_types = ['soc', 'soc']

        Shor_implies_feas = False

    elif inst["tipe"] == 'sqrt':

        A = np.eye(n)
        K_dims = [n]
        K_types = ['soc']

        g1 = inst["g1"]
        h1 = inst["h1"]

        for i in range(0, len(g1)):

            tmp2 = np.zeros((1, n))
            tmp2[0, 0] = 0.5
            tmp3 = np.hstack((g1[i, 0], h1[:, i].T))
            tmp4 = np.hstack((np.zeros((n-1, 1)), np.eye(n-1)))

            A = np.vstack((A, tmp2, tmp3, tmp4))
            K_dims = np.append(K_dims, n + 1)
            K_types = np.append(K_types, 'rsoc')

        Shor_implies_feas = True

    else:
        print("Error: tipe is invalid")
        exit()

    inst_std_form = {
        "n"                 : n,
        "Q"                 : Q,
        "A"                 : A,
        "K_dims"            : K_dims,
        "K_types"           : K_types,
        "x0"                : np.vstack(([[1]], inst["x0"])),
        "first_entry_1"     : True,
        "Shor_implies_feas" : Shor_implies_feas,
        "Shor_bounded"      : True
    }

    return inst_std_form


###############################################################################

def convert_lift_to_standard_form(lift):

    Q = lift["Qc"]

    n = lift["np2"]

    if lift["tipe"] == 'lin':

        tmp1 = lift["l0"].T
        tmp2 = lift["l1"].T
        tmp3 = lift["P"]
        A = np.vstack((tmp1, tmp2, tmp3))

        K_dims = [2, n - 1]

        K_types = ['nonneg', 'soc']

    elif lift["tipe"] == 'sqrt':

        l0 = lift["l0"]
        l1 = lift["l1"]

        K_dims = [1 + np.shape(l1)[1]]
        K_types = ['nonneg']
        A = lift["l0"].T
        for i in range(0, np.shape(l1)[1]):
            tmp = copy.copy(l1[:, i])
            tmp[n - 1] = -1
            #  l1[n - 1, i] = -1
            A = np.vstack((A, tmp.T))

        tmp2 = np.zeros((1, n))
        tmp2[0, 0] = 0.5
        A = np.vstack((A, tmp2, lift["P"]))
        K_dims = np.append(K_dims, n)
        K_types = np.append(K_types, 'rsoc')

    else:
        print("Error: tipe is invalid")
        exit()

    lift_std_form = {
        "n"                 : n,
        "Q"                 : Q,
        "A"                 : A,
        "K_dims"            : K_dims,
        "K_types"           : K_types,
        "x0"                : lift["w0"],
        "first_entry_1"     : True,
        "Shor_implies_feas" : False,
        "Shor_bounded"      : False # Not sure about this
    }

    return lift_std_form

###############################################################################

def lift_concentric_ttrs_instance(inst_std_form):

    n                 = inst_std_form["n"];
    Q                 = inst_std_form["Q"]
    A                 = inst_std_form["A"]    
    K_dims            = inst_std_form["K_dims"]
    K_types           = inst_std_form["K_types"]
    x0                = inst_std_form["x0"]
    first_entry_1     = inst_std_form["first_entry_1"]
    Shor_implies_feas = inst_std_form["Shor_implies_feas"]
    Shor_bounded      = inst_std_form["Shor_bounded"]


    tmp = np.diag(A[ (n+1):(2*n), 1:n])
    d = np.array([tmp**2]).T

    n = n - 1 # Now n is the original, original dimension

    e = np.ones((n, 1))

    Q = np.vstack((np.hstack((Q, np.zeros((n+1, n)))), np.zeros((n, 2*n + 1))))

    A1 = np.hstack(([[1]], np.zeros((1, n)), -e.T))
    A2 = np.hstack(([[1]], np.zeros((1, n)), -d.T))
    A = np.vstack((A1, A2))

    K_types = ['nonneg']
    K_dims = [2]

    for ind in range(0, n):

        tmp1 = np.zeros((1, 2*n + 1))
        tmp1[0, 0] = 0.5

        tmp2 = np.zeros((1, 2*n + 1))
        tmp2[0, 1 + n + ind] = 1

        tmp3 = np.zeros((1, 2*n + 1))
        tmp3[0, 1 + ind] = 1

        tmp = np.vstack((tmp1, tmp2, tmp3))

        A = np.vstack((A, tmp))

        K_types.append('rsoc')
        K_dims.append(3)

    n = np.shape(A)[1]

    lift = {
        "n"                 : n,
        "Q"                 : Q,
        "A"                 : A,
        "K_dims"            : K_dims,
        "K_types"           : K_types,
        "x0"                : -np.inf,
        "first_entry_1"     : True,
        "Shor_implies_feas" : True, # Not sure about this
        "Shor_bounded"      : True # Not sure about this
    }

    return lift
