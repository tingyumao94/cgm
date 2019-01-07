import numpy as np
from collections import Counter


# generate some random data from gaussian distribution
x = np.random.randn(1000)
x = (x[np.abs(x) < 1] * 100).astype('int')
c = Counter()

#