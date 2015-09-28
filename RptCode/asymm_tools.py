"""asymm_tools.py: Functions for calculations needed in strand symmetry modeling."""
#!/usr/bin/python
import threading
import scipy.linalg as sl
from numpy import *
from fp_check import *
from RptMatrix import *
from semiphore import *

############################################################
# MatCalcExcep: An exception to throw when a calculation isn't working right.
class MatCalcExcep(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return str(self.msg)

############################################################
# Usful matrix functions
def matrix2list(M):
    """Convert a matrix two a row-order list"""
    return list(M.flat)

def list2matrix(L, num_rows = -1):
    """Convery a list representing a row-major ordering of a matrix to a matrix.
       Assumed to be square if not otherwise specified."""
    M = matrix(L)
    return M.reshape(sqrt(len(L)) if num_rows == -1 else num_rows, -1)

def matrixSum(L, dtype = None):
    """Return the sum of a list of matrices"""
    S = zeros(L[0].shape, dtype if dtype else L[0].dtype)
    for M in L:
        S += M
    return S

def matrixAverage(L, dtype = None):
    """Take a list of same-sized matrices and return the average matrix"""
    return matrixSum(L, dtype) / len(L)

############################################################
# Functions for computing the different realted matrices
def compute_P_bad(C, asymm):
    """This is for debugging only -- do not use!!!!"""
    print("Warning: using compute_P_bad")
    P = vstack([C[i,:]/float64(sum(C[i,:])) if sum > 0 else zeros((1,4)) for i in range(4)])
    if not asymm:
        P2 = zeros((4,4), float64)
        for i in range(4):
            for j in range(4):
                if i != j:
                    P2[i,j] = 0.5*(P[i,j]+P[3-i,3-j]) 
            P2[i,i] = 1 - P2[i,].sum()
        P = P2
    return P

def compute_P(C, asymm):
    """Given a C matrix compute the P matrix.
    * C: The count-matrix.
    * asymm: boolean reflecting whether the model should be strand asymmetric.
    Raises MatCalcExcept if:
    * Matrix has a zero diagonal element.
    * Other computation error.
    """
    if not C.diagonal().all():
        raise MatCalcExcep("Count matrix has a zero diagonal (raised by compute_P).")

    if asymm:
        P = vstack([C[i,:]/float64(sum(C[i,:])) if sum(C[i,:]) > 0 else zeros((1,4)) for i in range(4)])
    else:
        P = zeros((4,4), float64)

        lambda1 = float64(C[0,0] + C[3,3] + C[0,1] + C[3,2] + C[0,2] + C[3,1] + C[0,3] + C[3,0])
        lambda2 = float64(C[1,0] + C[2,3] + C[1,1] + C[2,2] + C[1,2] + C[2,1] + C[1,3] + C[2,0])

        P[0,0] = (C[0,0] + C[3,3]) / lambda1
        P[0,1] = (C[0,1] + C[3,2]) / lambda1
        P[0,2] = (C[0,2] + C[3,1]) / lambda1
        P[0,3] = (C[0,3] + C[3,0]) / lambda1

        P[1,0] = (C[1,0] + C[2,3]) / lambda2
        P[1,1] = (C[1,1] + C[2,2]) / lambda2
        P[1,2] = (C[1,2] + C[2,1]) / lambda2
        P[1,3] = (C[1,3] + C[2,0]) / lambda2

        P[2,0] = P[1,3]
        P[2,1] = P[1,2]
        P[2,2] = P[1,1]
        P[2,3] = P[1,0]

        P[3,0] = P[0,3]
        P[3,1] = P[0,2]
        P[3,2] = P[0,1]
        P[3,3] = P[0,0]
        

        # lambda1 = float64(C[0,].sum() + C[3,].sum() - C[3,3])
        # lambda2 = float64(C[1,].sum() + C[2,].sum() - C[2,2])
        # if lambda1 == 0 or lambda2 == 0:
        #     print("If this is being invoked, something is potentially wrong.  asymm_tools.py: compute_P")
        #     raise MatCalcExcep("0-valued lambda values in symmatry model")

        # P[0,] = [ C[0,0] / lambda1, (C[0,1] + C[3,2]) / lambda1, (C[0,2] + C[3,1]) / lambda1, (C[0,3] + C[3,0]) / lambda1 ]
        # P[1,] = [ (C[1,0] + C[2,3]) / lambda2, C[1,1] / lambda2, (C[1,2] + C[2,1]) / lambda2, (C[1,3] + C[2,0]) / lambda2 ]
        # P[2,] = [ P[1,3-j] for j in range(4) ]
        # P[3,] = [ P[0,3-j] for j in range(4) ]
        

    return P    

def isP(M):
    """Return P if M is a legitimate discrete transition matrix"""
    return isOne(M.sum(1)).all() and gteZero(M).all() and gteZero(-1*M + 1).all()

def isQ(M):
    return isZero(M.sum(1)).all() and gteZero(-1*diag(M)).all() and gteZero(M - diag(M)*np.identity(4)).all()

sqrt2 = sqrt(2)
def compute_SIG(C):
    """Compute SIG matrix for Carin's code.  Only needed for asymmetric model.
       SIG[i,j] = sqrt(2*C[i,j])/sum_k(C[i,k])"""
    S = ones((4,4))
    for i in range(4):
        s = float64(sum(C[i,:]))
        if s == 0:
            continue
        for j in range(4):
            S[i,j] = sqrt2*sqrt(C[i,j])/s if C[i,j] != 0 else float64(1)
    return S
    

def compute_Rt(P):
    """Gen a P matrix, calculate the R*t matrix."""
    try:
        Rt = matrix(sl.logm(P, disp = False)[0])
    except Exception as E:
        raise MatCalcExcep(str(E))

    return makeReal(Rt)

def compute_d(M):
    """Computes the distance of M.  If M rows sum to 1, it assumes it is a probability matrix
    and uses log-det.  If they sum to 0, it assumes it is a rate matrix and computes 
    accordingly."""
    if isP(M):
        return -0.25*(linalg.slogdet(M)[1])
    elif isQ(M):
        return -0.25*sum(diagonal(M))
    else:
        raise MatCalcExcep("compute_d given a matrix that was not a correct probability or rate matrix")

def compute_q(M, d = None):
    """Compute the q matrix from the P or Rt matrix.  If d not provide, calculates it."""
    if d == None:
        d = compute_d(M)

    if isP(M):
        Rt = compute_Rt(M)
    elif isQ(M):
        Rt = M
    else:
        raise MatCalcExcep("compute_q given a matrix that was not a correct probability or rate matrix")

    if isZero(d) or d == inf:
        raise MatCalcExcep("Computing q for a zero- or inf-distance matrix (compute_q)")

    return Rt / d
                               
#def makeSym(q):   # Based on invalid model -- should not be used!!!
#    """Given an asymmetric q model, convert it to a symmetric q model"""
#    return matrix([[0.5*(q[i,j] + q[3-i,3-j]) for j in range(4)] for i in range(4)])

def compute_PfromRt(Rt):
    """Given an Rt matrix compute the corresponding P matrix."""
    return sl.expm(Rt)

def compute_logL(P, C):
    """Compute the log likelihood from a single P and C matrix pair"""
    #print(P)
    if not isOneV(P.sum(1)).all():    # This needs to be further investigated
        return 0
    return sum([C[i,j]*log(P[i,j]) for i in range(4) for j in range(4) if P[i,j] > 0])

def BIC(logL, free_params, data_magnitude):
    return -2*logL + free_params*log(data_magnitude)

def create_global_d(psm_files, output):
    """Calculate the global distance for each family and save to a file.   Assumes global strand-symmetry."""

    D = {}    # Map each alpha to a count matrix
    for file in psm_files:
        R = load_psm(file)
        for r in R:
            if r.class_name not in D:
                D[r.class_name] = zeros((4,4))
            D[r.class_name] += r.M
    
    with open(output, "w") as fp:
        for alpha in sorted(D.keys()):
            d = compute_d(compute_P(D[alpha], False))
            fp.write("{:<20}{:<20}\n".format(alpha,d))

def global_d(global_file):
    """Return a dictionary mapping family name to global distance, as recorded in file."""
    return {alpha:float64(v) for line in open(global_file) for alpha,v in [re.split("\s+", line.rstrip())]}

########################
# Threaded utility functions
######################
# Code for threaded calculation of matrices
def _calcMats(C, asymm = True):
    """Given a C, calculate P, d, and q"""
    try:
        P = compute_P(C, asymm)
        d = compute_d(P)
        q = compute_q(P,d)
        return P,d,q
    except MatCalcExcep:
        return None, None, None

class _calcMatsThread(threading.Thread):
    """Helper class for a threaded version of _calcMats"""
    def __init__(self, C, asymm):
        threading.Thread.__init__(self)
        self.C = C
        self.asymm = asymm

    def run(self):
        threadLimiter.acquire()
        try:
            self.mats = _calcMats(self.C, self.asymm)
        finally:
            threadLimiter.release()
        

    def get_mats(self):
        return self.mats

def calcMats(C_list, asymm, threaded = False):
    """Given a list of C matrices, return a list containing P/d/q tuples.
    Returns None for all three if any of them cannot be calculated."""
    if threaded and len(C_list) > 1:
        threads = [_calcMatsThread(C, asymm) for C in C_list]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return [t.get_mats() for t in threads]
    else:
        return [_calcMats(C, asymm) for C in C_list]

######
class _calcLHThread(threading.Thread):
    """Helper class for a the threaded calculation of log-likelhood"""
    def __init__(self, C, P, d = None):
        threading.Thread.__init__(self)
        self.C = C
        self.P = P

    def run(self):  # M must be P or q*d
        threadLimiter.acquire()
        try:
            self.LH = compute_logL(self.P,self.C)
        finally:
            threadLimiter.release()

    def get_LH(self):
        return self.LH

def calcLH(C_list, P_list, threaded = False):
    L = zip(C_list, P_list)
    if threaded:
        threads = [_calcLHThread(C, P) for C,P in L]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return [t.get_LH() for t in threads]
    else:
        return [compute_logL(P, C) for C,P in L]

            

########################
# The following are really written for debugging.
def calc_all(rpt_list, asymm = True):
    """Given a repeat list, augment each object with all associated matrices, and return a list of useful repeats and the averaged q.
    (Written for use in debugging.)"""

    S = {line.rstrip() for line in open("martin_repeats.txt")}
    R = []
    for r in rpt_list:
        if r.M.sum() < 40 or r.class_name not in S:
            continue
        
        try:
            P = compute_P(r.M, asymm)
            d = compute_d(P)
            if d == 0 or d == inf:
                continue
            q = makeRealV(compute_q(P, d))
            
            r.P = P
            r.d = d
            r.q = q
            R.append(r)
        except:
            continue
        
    
    Q = zeros((4,4))
    for r in R:
        Q += r.q
    Q = Q / len(R)

    return Q, R


def calc_for_gene(chr, start, finish, rpt_list = None):
    """Return a list of repeats specifically for a specified genes.
    rpt_list will be read in from the fuls .psm file if not already present."""
    if not rpt_list:
        print("Loading %s" % chr)
        rpt_list = load_psm("%s.psm" % chr)

    assert rpt_list[0].chr == chr

    i = 0
    while i < len(rpt_list) and rpt_list[i].start < start: i += 1
    j = i+1
    while j < len(rpt_list) and rpt_list[j].finish < finish: j += 1

    S = {line.rstrip() for line in open("martin_repeats.txt")}

    R = [r for r in rpt_list[i:j] if r.class_name in S]

    return R

