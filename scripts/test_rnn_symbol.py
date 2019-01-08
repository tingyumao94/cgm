import mxnet as mx
import numpy as np

batch_size = 32
time_steps = 10
num_hidden = 16

h0 = np.random.randn(batch_size, num_hidden)
c0 = np.random.randn(batch_size, num_hidden)
x = np.random.randn(batch_size, time_steps, num_hidden)
y = np.random.randn(batch_size, time_steps, num_hidden)

h_begin_states = mx.sym.Variable('h_begin_states')
c_begin_states = mx.sym.Variable('c_begin_states')
in_data = mx.sym.Variable('in_data')
out_data = mx.sym.Variable('out_data')

data_names = ['h_begin_states', 'c_begin_states', 'in_data']
data_shapes = [('h_begin_states', (batch_size, num_hidden)), ('c_begin_states', (batch_size, num_hidden)),
               ('in_features', (batch_size, time_steps, num_hidden))]
label_names = ['out_data']
label_shapes = [('out_data', (batch_size, time_steps, 1))]

