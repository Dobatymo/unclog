cimport cython
from cython.parallel cimport prange
from libc.math cimport log
cimport numpy as np

ctypedef fused my_type:
	float
	double

@cython.boundscheck(False)
@cython.wraparound(False)
cdef my_type _logtrace(my_type[:, ::1] arr) nogil:
	cdef Py_ssize_t X = arr.shape[0]
	cdef Py_ssize_t x
	cdef my_type out = 0.

	for x in range(X):
		out += log(arr[x, x])

	return out

def logtrace(my_type[:, ::1] arr):
	return _logtrace(arr)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _logtrace_batch(my_type[:, :, ::1] arr, my_type[::1] out) nogil:
	cdef Py_ssize_t B = arr.shape[0]
	cdef Py_ssize_t X = arr.shape[1]

	cdef Py_ssize_t b
	cdef Py_ssize_t x

	for b in prange(B, schedule="static"):
		out[b] = 0.
		for x in range(X):
			out[b] += log(arr[b, x, x])

def logtrace_batch(my_type[:, :, ::1] arr, my_type[::1] out):
	_logtrace_batch(arr, out)
