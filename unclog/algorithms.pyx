cimport cython

ctypedef fused reals:
	short
	int
	long
	float
	double

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _insertion(reals[::1] seq, Py_ssize_t left, Py_ssize_t right, Py_ssize_t gap) nogil:

	cdef Py_ssize_t loc = left + gap
	cdef Py_ssize_t i
	cdef reals value

	while loc <= right:
		i = loc - gap
		value = seq[loc]
		while i >= left and seq[i] > value:
			seq[i+gap] = seq[i]
			i -= gap
		seq[i+gap] = value
		loc += gap

cdef Py_ssize_t GROUP_SIZE = 5

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Py_ssize_t _median_of_medians(reals[::1] seq, Py_ssize_t left, Py_ssize_t right, Py_ssize_t gap) nogil:

	cdef Py_ssize_t span = GROUP_SIZE*gap
	cdef Py_ssize_t num = (right - left + 1) // span

	if num == 0:
		_insertion(seq, left, right, gap) # sort
		num = (right - left + 1) // gap
		return left + gap*(num-1) // 2 # pick median

	cdef Py_ssize_t s = left
	while s < right-span:
		_insertion(seq, s, s + span - 1, gap)
		s += span

	if num < GROUP_SIZE:
		_insertion(seq, left+span // 2, right, span)
		return left + num*span // 2
	else:
		return _median_of_medians(seq, left + span // 2, s-1, span)

def median_of_medians(reals[::1] seq, Py_ssize_t left=0, Py_ssize_t right=-1):

	""" Approximate median selection algorithm. """

	if right == -1:
		right = len(seq) - 1

	return _median_of_medians(seq, left, right, 1)
