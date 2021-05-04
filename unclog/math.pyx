cimport cython
from cython.parallel cimport prange
from libc.math cimport log
import numpy as np
cimport numpy as cnp

ctypedef fused my_float:
	float
	double

ctypedef fused my_int:
	int
	long

@cython.boundscheck(False)
@cython.wraparound(False)
cdef my_float _logtrace(my_float[:, ::1] arr) nogil:
	cdef Py_ssize_t X = arr.shape[0]
	cdef Py_ssize_t x
	cdef my_float out = 0.

	for x in range(X):
		out += log(arr[x, x])

	return out

def logtrace(my_float[:, ::1] arr):
	return _logtrace(arr)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _logtrace_batch(my_float[:, :, ::1] arr, my_float[::1] out) nogil:
	cdef Py_ssize_t B = arr.shape[0]
	cdef Py_ssize_t X = arr.shape[1]

	cdef Py_ssize_t b
	cdef Py_ssize_t x

	for b in prange(B, schedule="static"):
		out[b] = 0.
		for x in range(X):
			out[b] += log(arr[b, x, x])

def logtrace_batch(my_float[:, :, ::1] arr, my_float[::1] out):
	_logtrace_batch(arr, out)

cdef my_int _max(my_int[:, ::1] arr):
	cdef Py_ssize_t B = arr.shape[0]
	cdef Py_ssize_t X = arr.shape[1]

	cdef Py_ssize_t b
	cdef Py_ssize_t x

	cdef my_int ret = 0

	for b in range(B):
		for x in range(X):
			ret = max(ret, arr[b, x])

	return ret

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _bincount_batch(my_int[:, ::1] arr, my_int[:, ::1] out) nogil:
	cdef Py_ssize_t B = arr.shape[0]
	cdef Py_ssize_t X = arr.shape[1]

	cdef Py_ssize_t b
	cdef Py_ssize_t x

	for b in prange(B, schedule="static"):
		for x in range(X):
			out[b, arr[b, x]] += 1

def bincount_batch(my_int[:, ::1] arr, int axis=-1, int minlength=0):
	cdef int B = arr.shape[0]
	cdef int N = max(minlength, _max(arr) + 1)

	if axis != -1:
		raise ValueError("Only bincount in the last axis is supported")

	cdef my_int[:, ::1] out = np.zeros((B, N), dtype=np.int)

	_bincount_batch(arr, out)

	return out
