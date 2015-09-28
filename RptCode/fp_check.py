"""
fp_check.py:
Functions for comparing floating-point values, using a tolerance (default value 0.0001) to
define euqality.  (e.g. x==y if abs(x - y) < tolerance0.

For each function there is a vectorized version, which applies the function to each element 
of a matric, returning a matric of the results.
"""

import sys
import math
from numpy import matrix
from numpy import vectorize

_tolerance = 0.00001
def set_tolerance(t):
    """Change the tolerance constant (default = 0.00001)"""
    global _tolerance
    _tolerance = t

def isZero(a):
    """Determine if a floating point is 0 within a floating-point _tolerance factor"""
    return abs(a) < _tolerance

_isZeroV = vectorize(isZero)
def isZeroV(M):
    """Vectorized version of isZero."""
    return _isZeroV(M)

def isOne(a):
    """Check if a floating point is 1 within a floating-point _tolerance factor"""
    return abs(a-1) < _tolerance

_isOneV = vectorize(isOne)
def isOneV(M):
    """Vectorized version of isOne."""
    return _isOneV(M)

def gtZero(a):
    """Determine if a is greater than 0 with a floating-point _tolerance factor"""
    return a >= _tolerance

_gtZeroV = vectorize(gtZero)
def gtZeroV(M):
    """Vectorized version of gtZero"""
    return _gtZeroV(M)

def gteZero(a):
    """Determine if a is >= 0 with a a floating-point _tolerance factor"""
    return a >= -_tolerance


def isReal(a):
    """Determine if a floating point is a real number within a floaing-point _tolerance factor"""
    return abs(a.imag) < _tolerance

_isRealV = vectorize(isReal)
def isRealV(M):
    """Vectorized version of isReal"""
    return _isRealV(M)


def makeReal(a):
    """Convert a compex number to a real by truncating the imaginary portion."""
    return a.real

_makeRealV = vectorize(makeReal)
def makeRealV(M):
    """Vectorized version of makeReal"""
    return _makeRealV(M)

def isEqual(a, b):
    """Determine if two floating points are equal within a floating-point _tolerance factor"""
    return abs(a-b) < _tolerance

def isEqualV(a,b):
    """Vectorized version of isEqual"""
    return _isZero(a-b)

