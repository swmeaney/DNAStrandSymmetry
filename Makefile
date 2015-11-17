CC = g++

symmetryTester: symmetryTester.cpp
	g++ -O3 -std=c++11 -o main symmetryTester.cpp -I /software/armadillo/4.450.4/include -DARMA_USE_BLAS -DARMA_USE_LAPACK -DARMA_DONT_USE_WRAPPER /software/lapack/lapack-3.5.0/liblapack.a  /software/blas/libblas.a  -lgfortran
