from unclog import logtrace

import numpy as np

def create_typed(dtype):
	arr = np.random.uniform(0, 10, (100, 100, 100)).astype(dtype)
	out = np.empty(100, dtype=dtype)

	return arr, out

arr, out = create_typed(np.float32)
logtrace(arr, out)

arr, out = create_typed(np.float64)
logtrace(arr, out)
