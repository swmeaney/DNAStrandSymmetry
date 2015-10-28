"""check_asymm.py: Impelmentation of multiple algorithms for checking asymmetry."""

from fp_check import *
from asymm_tools import *
from RptMatrix import *
from numpy import *
from scipy.linalg import expm
import subprocess
import re
import sys
import argparse


############################################################
# Algorithms for checking asymmetry of a partitions and zero or more optional parameters.
# The list of repeats is assumed to have already been filtered -- all will be used
# unless they result in a dismiss-able numercial error (e.g. a singular matrix or distance
# of 0).
# Each algorithm returns a five element list containing:
# [0] A filtered subset of rpt_list.
# [1] BIC value (symmetric model)
# [2] BIC value (asymmetric model)
# [3] Bool: Is BIC assymetric < BIC symmetric - 2?
# [4] the q matrix



####################
# 
def create_mlw_file(q, P_list, S_list, d_list, tmp_file):
    """Create a mlw file used to pass information from this code to the minLsei_wrapper executable."""
    #### Here is what we need to "pass" to the code.
    #### WARNING: Martin's code is expecting many of these to start at 1 instead of 0. 
    ####          (That is, the array [1,1,1,1] would instead be [0,1,1,1,1].)  This wil
    ###           need to be accounted for.
    k = len(P_list)
    q = list(q.flat)
    y_nr = [v for M in P_list for v in M.flat]
    sig_nr = [v for M in S_list for v in M.flat]
    d_alpha = d_list
    

    # Now -- write to file and invoke minLsei_wrapper.  Would eventually like to make this a direct call 
    # passing k, q, y_nr, sig_nr, and d_alpha.
    with open(tmp_file, "w") as wp:
        wp.write("k\t" + str(k) + "\n")
        wp.write("q\t" + " ".join([str(v) for v in q]) + "\n")
        wp.write("y_nr\t" + " ".join([str(v) for v in y_nr]) + "\n")
        wp.write("sig_nr\t" + " ".join([str(v) for v in sig_nr]) + "\n")
        wp.write("d_alpha\t" + " ".join([str(v) for v in d_alpha]) + "\n")    

def run_minLsei_wrapper(q, P_list, S_list, d_list, tmp_file):
    create_mlw_file(q, P_list, S_list, d_list, tmp_file)
    p = subprocess.Popen("./minLsei_wrapper %s" % (tmp_file), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)

    # Retrieve results.  
    o,e = p.communicate()
    #print(o.encode())

    if e:
        return None,None

    try:
        symm_line, asymm_line = re.split("\n", o.encode().rstrip());
    except:
        sys.stderr.write("Error in return value: %s\n" % (e.encode().rstrip()))
        return None, None

    if symm_line == "-1":
        return None, None
    else:
        i = symm_line.find(" ")
        j = symm_line.find("*")
        L_symm = float(symm_line[:i])
        BIC_symm = BIC(L_symm, 5 + len(P_list), 16*len(P_list))
        #q_symm = matrix(symm_line[i+1:j-1])
        #d_symm = [float(i) for i in re.split("\s+", symm_line[j+2:].strip())]
        #logLh_symm = sum([compute_logL(expm(q_symm*d),r.M) for d,r in zip(d_symm, R_list)])
        #BIC_symm = BIC(logLh_symm, 5 + len(R_list), 16*len(R_list))
        #print(L_symm)

    if asymm_line == "-1":
        return None, None
    else:
        i = asymm_line.find(" ")
        j = asymm_line.find("*")
        L_asymm = float(asymm_line[:i])
        BIC_asymm = BIC(L_asymm, 11 + len(P_list), 16*len(P_list))
        #q_asymm = matrix(asymm_line[i+1:j-1])
        #d_asymm = [float(i) for i in re.split("\s+", asymm_line[j+2:].strip())]
        #logLh_asymm = sum([compute_logL(expm(q_asymm*d),r.M) for d,r in zip(d_asymm, R_list)])
        #BIC_asymm = BIC(logLh_asymm, 11 + len(R_list), 16*len(R_list))
        #print(L_asymm)

    return BIC_symm,BIC_asymm

####################
def Mugal1(rpt_list, tmp_file, global_dist_file = None):  # Ignore global_dist_file
    """Using minLsei + BIC.  Initital estimates are averaged q and instance d."""
    # First: compute the y_nr and sig_nr
    P_list = []      # List of P matrices
    S_list = []      # List of SIG matrices
    d_list = []      # d value for each repeat
    q_list = []      # List of q matrices
    R_list = []      # List of repeats after excluding those that were tossed
    for r in rpt_list:
        try:
            P = compute_P(r.M, True)
            S = compute_SIG(r.M)
            d = compute_d(P)
            q = compute_q(P,d)
            P_list.append(P)
            S_list.append(S)
            d_list.append(d)
            q_list.append(q)
            R_list.append(r)
        except MatCalcExcep:
            continue

    q = matrixAverage([makeRealV(q) for q in q_list])

    BIC_symm,BIC_asymm = run_minLsei_wrapper(q, P_list, S_list, d_list, tmp_file)

    isAsymm = BIC_asymm is not None  and BIC_symm is not None and BIC_asymm < BIC_symm - 2
    return R_list, BIC_symm, BIC_asymm, isAsymm, None

####################
def Mugal2(rpt_list, global_dist_file, tmp_file):
    """Using minLsei + BIC.  Initital estimates are averaged q and family global d."""

    D = global_d(global_dist_file)

    # First: compute the y_nr and sig_nr
    P_list = []      # List of P matrices
    S_list = []      # List of SIG matrices
    d_list = []      # d value for each repeat
    q_list = []      # List of q matrices
    R_list = []      # List of repeats after excluding those that were tossed
    for r in rpt_list:
        try:
            if r.class_name not in D:
                continue
            P = compute_P(r.M, True)
            S = compute_SIG(r.M)
            d = compute_d(P)
            q = compute_q(P,d)
            P_list.append(P)
            S_list.append(S)
            d_list.append(D[r.class_name])   # Here we insert the global distance instead of the local distance.
            q_list.append(q)
            R_list.append(r)
        except MatCalcExcep:
            continue

    q = matrixAverage([makeRealV(q) for q in q_list])

    create_mlw_file(R_list, q, P_list, S_list, d_list, tmp_file)
    p = subprocess.Popen("./minLsei_wrapper %s" % (tmp_file), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)

    # Retrieve results.  
    o,e = p.communicate()

    if e:
        return None, None, None, None, None


    try:
        symm_line, asymm_line = re.split("\n", o.encode().rstrip());
    except:
        sys.stderr.write("Error in return value: %s" % (e.encode().rstrip()))


    if symm_line == "-1":
        L_symm, q_symm, logLh_symm, BIC_symm = -1, None, -1, None
    else:
        i = symm_line.find(" ")
        j = symm_line.find("*")
        L_symm = float(symm_line[:i])
        q_symm = matrix(symm_line[i+1:j-1])
        d_symm = [float(i) for i in re.split("\s+", symm_line[j+2:].strip())]
        logLh_symm = sum([compute_logL(expm(q_symm*d),r.M) for d,r in zip(d_symm, R_list)])
        BIC_symm = BIC(logLh_symm, 5 + len(R_list), 16*len(R_list))

    if asymm_line == "-1":
        L_asymm, q_asymm, logLh_asymm, BIC_asymm= -1, None, -1, None
    else:
        i = asymm_line.find(" ")
        j = asymm_line.find("*")
        L_asymm = float(asymm_line[:i])
        q_asymm = matrix(asymm_line[i+1:j-1])
        d_asymm = [float(i) for i in re.split("\s+", asymm_line[j+2:].strip())]
        logLh_asymm = sum([compute_logL(expm(q_asymm*d),r.M) for d,r in zip(d_asymm, R_list)])
        BIC_asymm = BIC(logLh_asymm, 11 + len(R_list), 16*len(R_list))


    isAsymm = BIC_asymm is not None  and BIC_symm is not None and BIC_asymm < BIC_symm - 2
    return R_list, BIC_symm, BIC_asymm, isAsymm, q_asymm if isAsymm else q_symm

# ####################
# # Alg. 1: Uses Mugal's code.  # FORGET THIS ALG -- IT DOESNT WORK
# def Mugal3(rpt_list, tmp_file, global_dist_file = None):
#     """Using minLsei, with interval formulations.  Initial estimates are the averaged q and instance d."""

#     # First: compute the y_nr and sig_nr
#     P_list = []      # List of P matrices
#     S_list = []      # List of SIG matrices
#     d_list = []      # d value for each repeat
#     q_list = []      # List of q matrices
#     R_list = []      # List of repeats after excluding those that were tossed
#     for r in rpt_list:
#         try:
#             P = compute_P(r.M, True)
#             S = compute_SIG(r.M)
#             d = compute_d(P)
#             q = compute_q(P,d)
#             P_list.append(P)
#             S_list.append(S)
#             d_list.append(d) 
#             q_list.append(q)
#             R_list.append(r)
#         except MatCalcExcep:
#             continue

#     q = sum(makeRealV(q) for q in q_list)/len(q_list)

#     create_mlw_file(R_list, q, P_list, S_list, d_list, tmp_file)
#     p = subprocess.Popen("./minLsei_wrapper2 %s" % (tmp_file), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)

#     # Retrieve results.  
#     o,e = p.communicate()

#     if e:
#         return None, None, None, None, None


#     try:
#         output = o.encode().rstrip()
#     except:
#         sys.stderr.write("Error in return value: %s" % (e.encode().rstrip()))

#     if output == "-1":
#         return None, None, None, None, None
#     else:
#         num_asymm = 0
#         for line in output.rstrip().split("\n"):
#             A = re.split("\t", line.rstrip())
#             lb = float(A[1]);
#             ub = float(A[2]);
#             if gtZero(lb) or gtZero(-gb):
#                 num_asymm += 1
        
#     return R_list, None, None, num_asymm > 0

####################
# Alg. 1: Uses Mugal's code.   FORGET THIS ALG -- IT DOESN'T WORK
# def Mugal4(rpt_list, global_dist_file, tmp_file):
#     """Using minLsei, with interval formulations.  Initial estimates are the averaged q and global d"""

#     D = global_d(global_dist_file)

#     # First: compute the y_nr and sig_nr
#     P_list = []      # List of P matrices
#     S_list = []      # List of SIG matrices
#     d_list = []      # d value for each repeat
#     q_list = []      # List of q matrices
#     R_list = []      # List of repeats after excluding those that were tossed
#     for r in rpt_list:
#         try:
#             if r.class_name not in D:
#                 continue
#             P = compute_P(r.M, True)
#             S = compute_SIG(r.M)
#             d = compute_d(P)
#             q = compute_q(P,d)
#             P_list.append(P)
#             S_list.append(S)
#             d_list.append(D[r.class_name]) 
#             q_list.append(q)
#             R_list.append(r)
#         except MatCalcExcep:
#             continue

#     q = sum(makeRealV(q) for q in q_list)/len(q_list)

#     create_mlw_file(R_list, q, P_list, S_list, d_list, tmp_file)
#     p = subprocess.Popen("./minLsei_wrapper2 %s" % (tmp_file), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)

#     # Retrieve results.  
#     o,e = p.communicate()

#     if e:
#         return -1, None, -1, None


#     try:
#         output = o.encode().rstrip()
#     except:
#         sys.stderr.write("Error in return value: %s" % (e.encode().rstrip()))

#     if output == "-1":
#         return None, None, None, None, None
#     else:
#         num_asymm = 0
#         for line in output.rstrip().split("\n"):
#             A = re.split("\t", line.rstrip())
#             lb = float(A[1]);
#             ub = float(A[2]);
#             if gtZero(lb) or gtZero(-gb):
#                 num_asymm += 1
        
#     return R_list, None, None, num_asymm > 0

####################
# def Mugal5(rpt_list, tmp_file, global_dist_file = None):
#     """Using minLsei + BIC.  Initital estimates are averaged q and instance d."""
#     # First: compute the y_nr and sig_nr
#     P_list = []      # List of P matrices
#     S_list = []      # List of SIG matrices
#     d_list = []      # d value for each repeat
#     q_list = []      # List of q matrices
#     R_list = []      # List of repeats after excluding those that were tossed
#     for r in rpt_list:
#         try:
#             P = compute_P(r.M, True)
#             S = compute_SIG(r.M)
#             d = compute_d(P)
#             q = compute_q(P,d)
#             P_list.append(P)
#             S_list.append(S)
#             d_list.append(d)
#             q_list.append(q)
#             R_list.append(r)
#         except MatCalcExcep:
#             continue

#     q = sum(makeRealV(q) for q in q_list)/len(q_list)

#     create_mlw_file(R_list, q, P_list, S_list, d_list, tmp_file)
#     p = subprocess.Popen("./minLsei_wrapper %s" % (tmp_file), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)

#     # Retrieve results.  
#     o,e = p.communicate()

#     if e:
#         return None, None, None, None, None

#     try:
#         symm_line, asymm_line = re.split("\n", o.encode().rstrip());
#     except:
#         sys.stderr.write("Error in return value: %s\n" % (e.encode().rstrip()))
#         return None, None, None, None, None

#     if symm_line == "-1":
#         L_symm, q_symm, logLh_symm, BIC_symm = -1, None, -1, None
#     else:        
#         i = symm_line.find(" ")
#         j = symm_line.find("*")
#         L_symm = float(symm_line[:i])
#         q_symm = matrix(symm_line[i+1:j-1])
#         d_symm = [float(i) for i in re.split("\s+", symm_line[j+2:].strip())]
#         BIC_symm = L_symm + 0.5*(len(R_list)+5)*log(16*len(R_list))

#     if asymm_line == "-1":
#         L_asymm, q_asymm, logLh_asymm, BIC_asymm= -1, None, -1, None
#     else:
#         i = asymm_line.find(" ")
#         j = asymm_line.find("*")
#         L_asymm = float(asymm_line[:i])
#         q_asymm = matrix(asymm_line[i+1:j-1])
#         d_asymm = [float(i) for i in re.split("\s+", asymm_line[j+2:].strip())]
#         BIC_asymm = L_asymm + 0.5*(len(R_list)+11)*log(16*len(R_list))

#     isAsymm = BIC_asymm is not None  and BIC_asymm is not None and BIC_asymm < BIC_symm -2
#     return R_list, BIC_symm, BIC_asymm, isAsymm, q_asymm if isAsymm else q_symm


##############################
def FHK1(rpt_list, global_dist_file = None, tmp_file = None):  
    """Calculate the fit by computing a q/d value for each instance, 
    then sum the log-likelihoods and sum the BIC values.
    q: Estimated by averaging the q for each instance.
    d: Estimated using the d for each instance.
    (global_dist_file and tmp_file parameters are ignored).
    """
    q_symm_list = []
    d_symm_list = []
    q_asymm_list = []
    d_asymm_list = []
    R_list = []
    for r in rpt_list:
        try:
            P_symm = compute_P(r.M, False)
            P_asymm = compute_P(r.M, True)
            d_symm = compute_d(P_symm)
            d_asymm = compute_d(P_asymm)
            q_symm = compute_q(P_symm, d_symm)
            q_asymm = compute_q(P_asymm, d_asymm)
            q_symm_list.append(q_symm)
            q_asymm_list.append(q_asymm)
            d_symm_list.append(d_symm)
            d_asymm_list.append(d_asymm)
            R_list.append(r)
        except MatCalcExcep:
            continue

    q_symm = matrixAverage([makeRealV(M) for M in q_symm_list])
    q_asymm = matrixAverage([makeRealV(M) for M in q_asymm_list])

    logLh_symm  = sum([compute_logL(expm(q_symm*d), r.M) for d,r in zip(d_symm_list, R_list)])
    logLh_asymm = sum([compute_logL(expm(q_asymm*d), r.M) for d,r in zip(d_asymm_list, R_list)])

    
    BIC_symm = BIC(logLh_symm, 5 + len(R_list), 16*len(R_list))
    BIC_asymm = BIC(logLh_asymm, 11 + len(R_list), 16*len(R_list))


    isAsymm = BIC_symm is not None and BIC_asymm is not None and BIC_asymm < BIC_symm - 2

    return R_list, BIC_symm, BIC_asymm, isAsymm, q_asymm if isAsymm else q_symm

##############################
def FHK2(rpt_list, global_dist_file, tmp_file = None): # tmp_file not needed
    """Calculate the fit by computing a q/d value for each instance and from that the BIC values.
    q: Estimated by averaging the q for each instance.
    d: Estimated using the global d for each instance.
    """

    D = global_d(global_dist_file)    

    q_symm_list = []
    d_symm_list = []
    q_asymm_list = []
    d_asymm_list = []
    R_list = []
    for r in rpt_list:
        try:
            if r.class_name not in D:
                continue
            P_symm = compute_P(r.M, False)
            P_asymm = compute_P(r.M, True)
            d_symm = compute_d(P_symm)
            d_asymm = compute_d(P_asymm)
            q_symm = compute_q(P_symm, d_symm)
            q_asymm = compute_q(P_asymm, d_asymm)
            q_symm_list.append(q_symm)
            q_asymm_list.append(q_asymm)
            d_symm_list.append(D[r.class_name])
            d_asymm_list.append(D[r.class_name])
            R_list.append(r)
        except MatCalcExcep as M:
            continue

    q_symm = matrixAverage([makeRealV(M) for M in q_symm_list])
    q_asymm = matrixAverage([makeRealV(M) for M in q_asymm_list])

    logLh_symm  = sum([compute_logL(expm(q_symm*d), r.M) for d,r in zip(d_symm_list, R_list)])
    logLh_asymm = sum([compute_logL(expm(q_asymm*d), r.M) for d,r in zip(d_asymm_list, R_list)])

    BIC_symm = BIC(logLh_symm, 5 + len(R_list), 16*len(R_list))
    BIC_asymm = BIC(logLh_asymm, 11 + len(R_list), 16*len(R_list))

    isAsymm = BIC_symm is not None  and BIC_asymm is not None and BIC_asymm < BIC_symm -2
    return R_list, BIC_symm, BIC_asymm, isAsymm, q_asymm if isAsymm else q_symm

##############################
def FHK3(rpt_list, global_dist_file = None, tmp_file = None):
    """Calculate the fit by computing q and d values by aggrigrated family.
    q: Estimated by averaging the q over the families.
    d: Estimated for a family using the average over the instances of the family.
    """

    # First: Create a list of fake repeats -- one for each family, where the C
    # matrices are combined.
    C_dic = {}
    for r in rpt_list:
        alpha = r.class_name
        if alpha not in C_dic:
            new_r = RptMatrix(None, None)
            new_r.class_name = alpha
            C_dic[alpha] = new_r
        C_dic[alpha].M += r.M
    family_rpt_list = C_dic.values()

    # Now we call algorithm 2 on the new repeat list
    return FHK1(family_rpt_list)

##############################
def FHK4(rpt_list, global_dist_file, tmp_file = None):
    """Calculate the fit by computing q and d values by aggrigrated family.
    q: Estimated by averaging the q over the families.
    d: Estimated for a family using the global for the family.
    """

    # First: Create a list of fake repeats -- one for each family, where the C
    # matrices are combined.
    C_dic = {}
    for r in rpt_list:
        alpha = r.class_name
        if alpha not in C_dic:
            new_r = RptMatrix(None, None)
            new_r.class_name = alpha
            C_dic[alpha] = new_r
        C_dic[alpha].M += r.M

    family_rpt_list = C_dic.values()

    # Now we call algorithm 2 on the new repeat list
    return FHK2(family_rpt_list, global_dist_file)

#############################
def FHK5(rpt_list, global_dist_file, tmp_file = None):
    """Estimated using one gene-wide P matrix calculated from the averaged q and the gene-global d."""
    
    q_symm_list  = []
    d_symm_list  = []
    q_asymm_list = []
    d_asymm_list = []
    R_list = []
    for r in rpt_list:
        try:
            P_symm = compute_P(r.M, False)
            P_asymm = compute_P(r.M, True)
            d_symm = compute_d(P_symm)
            d_asymm = compute_d(P_asymm)
            q_symm = compute_q(P_symm, d_symm)
            q_asymm = compute_q(P_asymm, d_asymm)
            q_symm_list.append(q_symm)
            q_asymm_list.append(q_asymm)
            d_symm_list.append(d_symm)
            d_asymm_list.append(d_asymm)
            R_list.append(r)
        except MatCalcExcep as M:
            continue    

    d_symm = mean(d_symm_list)
    d_asymm = mean(d_asymm_list)

    q_symm = matrixAverage([makeRealV(M) for M in q_symm_list])
    q_asymm = matrixAverage([makeRealV(M) for M in q_asymm_list])

    C = matrixSum([r.M for r in R_list])
                
    logLh_symm = compute_logL(expm(q_symm)*d_symm,C)
    logLh_asymm = compute_logL(expm(q_asymm)*d_asymm,C)

    BIC_symm = BIC(logLh_symm, 5 + len(R_list), 16*len(R_list))
    BIC_asymm = BIC(logLh_asymm, 11 + len(R_list), 16*len(R_list))

    isAsymm = BIC_symm is not None  and BIC_asymm is not None and BIC_asymm < BIC_symm -2
    return R_list, BIC_symm, BIC_asymm, isAsymm, q_asymm if isAsymm else q_symm

#############################
def FHK6(rpt_list, global_dist_file, tmp_file = None):
    """Estimated using one gene-wide P matrix calculated from the averaged q and the chromosome-global d."""
    
    D = global_d(global_dist_file)

    # SWM 2015-10-21 set type to list for python 3
    d_global = mean(list(D.values()))    

    q_symm_list = []
    d_symm_list = []
    q_asymm_list = []
    d_asymm_list = []
    R_list = []
    for r in rpt_list:
        try:
            if r.class_name not in D:
                continue
            P_symm = compute_P(r.M, False)
            P_asymm = compute_P(r.M, True)
            d_symm = compute_d(P_symm)
            d_asymm = compute_d(P_asymm)
            q_symm = compute_q(P_symm, d_symm)
            q_asymm = compute_q(P_asymm, d_asymm)
            q_symm_list.append(q_symm)
            q_asymm_list.append(q_asymm)
            d_symm_list.append(d_symm)
            d_asymm_list.append(d_asymm)
            R_list.append(r)
        except MatCalcExcep as M:
            continue    

    q_symm = matrixAverage([makeRealV(M) for M in q_symm_list])
    q_asymm = matrixAverage([makeRealV(M) for M in q_asymm_list])

    C = matrixSum([r.M for r in R_list])
                
    logLh_symm = compute_logL(expm(q_symm)*d_global,C)
    logLh_asymm = compute_logL(expm(q_asymm)*d_global,C)

    BIC_symm = BIC(logLh_symm, 5 + len(R_list), 16*len(R_list))
    BIC_asymm = BIC(logLh_asymm, 11 + len(R_list), 16*len(R_list))

    isAsymm = BIC_symm is not None  and BIC_asymm is not None and BIC_asymm < BIC_symm - 2
    return R_list, BIC_symm, BIC_asymm, isAsymm, q_asymm if isAsymm else q_symm


############################################################
# For testing a file.  Must be passed a generator associated with the file format.
# At each iteration, the generator should yield a tople consiting of:
# [0] partition identification information (e.g. gene name) -- string
# [1] parition chromosome -- string
# [2] partition start position -- int
# [3] partition finish position -- int
# [4] partition strand (+, C, or ?)
# [5-?] Anything else (optional)
def test_partitions(G, asymm_alg, tmp_file, fp = sys.stdout, rfp = sys.stdout):
    current_chr = None
    for T in G:

        name, chr_name = T[0:2]
        partition_start = int(T[2])
        partition_finish = int(T[3])
        strand = T[4]
        other_info = T[5:]

        if chr_name != current_chr:
            print("Loading: " + chr_name + ".psm")
            rpt_list = load_psm(chr_name + ".psm")
            i = 0
            current_chr = chr_name

        while i < len(rpt_list) and rpt_list[i].start < partition_start:   # Currently only allowing partitions compeletely contained
            i += 1

        j = i + 1
        while j < len(rpt_list) and rpt_list[j].finish < partition_finish:
            j = j + 1

        if j > i:
            print("%s %d %d" % (chr_name, partition_start, partition_finish))
            R = rpt_list[i:j]
            total_bases = sum([r.M.sum() for r in R])

            T = asymm_alg(R, tmp_file)

            s = "{part_name:<15}{chr_name:<10}{start:<15}{finish:<15}{strand:<5}{num_bases:<10}".format(part_name = name, chr_name = chr_name, start = partition_start, finish = partition_finish, strand = strand, num_bases = total_bases)
            s += "".join(["{<:15}".format(x) for x in other_info])
            s += "{:<15}".format(str(T[3]))
            fp.write(s + "\n")

            s = "{chr_name:<10}{start:<15}{finish:<15}".format(chr_name=chr_name, start=partition_start, finish=partition_finish)
            # SWM 2015-10-15 - added forced conversion to string after field name
            s += "{BIC_symm!s:<15}{BIC_asymm!s:<15}{asymm!s:<10}".format(BIC_symm=T[1], BIC_asymm=T[2], asymm = str(T[-2]))
            s += " ".join([str(v) for v in T[-1].flat]) if T[-1] is not None else "ERROR"
            (fp if T[-2] else rfp).write(s + "\n")
                
                         
############################################################                               
# Generation functions for different format
# Takes a file name and an optinal limiting chromsome name.
def carina_generator(file = "hg18_genes.rates.dat", limiting_chr = None, header = True):
    """For files in the format given to use by Carin (.dat)"""
    fp = open(file)
    if header:
        fp.readline()
    for line in fp:
        chr, start, finish = re.split("\s+", line)[:3]
        if not limiting_chr or chr == limiting_chr:
            yield "Unknown", chr, int(start), int(finish), '?'


############################################################
alg_set = {'Mugal1', 'Mugal3', 'Mugal5', 'FHK1', 'FHK3'}
def run_test(gene_file, alg, generator, tmp_file, header = False, dist_file = None, limiting_chr = None, output_file = None, reject_file = None):
    assert dist_file or (alg in alg_set), "Algorithm %s requires glboal distance file" % (alg)

    G = eval(generator)(gene_file, limiting_chr, header)
    f = eval(alg) if alg in alg_set else lambda R, x: eval(alg)(R, dist_file, tmp_file)
    ofp = open(output_file, "w") if (output_file and output_file != '-') else sys.stdout
    rfp = open(reject_file, "w") if (reject_file and reject_file != '-') else sys.stdout
    test_partitions(G, f, tmp_file, ofp, rfp)
    

####### Random functions for testing -- should eventually be removed
def test_load():
    global chr22
    global R

    chr22 = load_psm('chr22.psm')
    R = rptsInPartition(chr22, 19391990, 19523019)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Search for asymmetric genes")
    
    alg_parser = parser.add_argument_group("Algorithm choice")
    alg_group = alg_parser.add_mutually_exclusive_group()
    alg_group.add_argument("--M1", dest = 'alg', action = 'store_const', const = 'Mugal1', help = "Mugal algorithm: BIC, by instance distance (default)", default = "Mugal1")
    alg_group.add_argument("--M2", dest = 'alg', action = 'store_const', const = 'Mugal2', help = "Mugal algorithm: BIC, by global distance")
    alg_group.add_argument("--M3", dest = 'alg', action = 'store_const', const = 'Mugal3', help = "Mugal algorithm: intervals, by instance distamce")
    alg_group.add_argument("--M4", dest = 'alg', action = 'store_const', const = 'Mugal4', help = "Mugal algorithm: intervals, global distance")
    alg_group.add_argument("--M5", dest = 'alg', action = 'store_const', const = 'Mugal5', help = "Mugal algorithm: exactly as implemented in her code -- strand BIC")
    alg_group.add_argument("--F1", dest = 'alg', action = 'store_const', const = 'FHK1', help = "FHK algorithm: BIC, by instance distance")
    alg_group.add_argument("--F2", dest = 'alg', action = 'store_const', const = 'FHK2', help = "FHK algorithm: BIC, by global distance")
    alg_group.add_argument("--F3", dest = 'alg', action = 'store_const', const = 'FHK3', help = "FHK algorithm: BIC, grouped, by local family distance")
    alg_group.add_argument("--F4", dest = 'alg', action = 'store_const', const = 'FHK4', help = "FHK algorithm: BIC, grouped, by global family distance")
    alg_group.add_argument("--F5", dest = 'alg', action = 'store_const', const = 'FHK5', help = "FHK algorithm: BIC, single P, gene-global d")
    alg_group.add_argument("--F6", dest = 'alg', action = 'store_const', const = 'FHK6', help = "FHK algorithm: BIC, single P, genome-global d")


    format_parser = parser.add_argument_group("Gene data format")
    format_group = format_parser.add_mutually_exclusive_group()
    format_group.add_argument("-d", "--dat", dest = 'gene_format', action = 'store_const', const = 'carina_generator', help = "DAT forma (chr, start, stop...", default = "carina_generator")


    options_group = parser.add_argument_group(description = "Other options")
    options_group.add_argument("-H", dest = 'header', action = "store_false", help = "Gene file contains a header line", default = True)
    options_group.add_argument("-g", '--global_dist', dest = 'global_dist', help = "File of global distances", default = "hg18_global.txt")
    options_group.add_argument("-c", '--chr', dest = 'limiting_chr', help = "Limit to a single chromosome", default = None)
    options_group.add_argument("-t", "--tmp_file", dest = 'tmp_file', help = "Name of temp. file for passing data", default = "mlw.txt")

    parser.add_argument("gene_file", help = "File containing gene information")
    parser.add_argument("output_file", help = "Output file name (- for stdout)")
    parser.add_argument("reject_file", help = "Rejected file (- for stdout)")

    args = parser.parse_args()

    run_test(gene_file = args.gene_file, alg = args.alg, generator = args.gene_format, header = args.header, dist_file = args.global_dist, limiting_chr = args.limiting_chr, output_file = args.output_file, reject_file = args.reject_file, tmp_file = args.tmp_file)
