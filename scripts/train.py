import yaml
import mxnet as mx

from cgm.loader import CgmLoader

with open('./config/train_cgm_lstm.yaml') as f:
    cfg = yaml.load(f)

# load data
test_data = CgmLoader(subject_id=3, config=cfg, is_train=True)
val_data = CgmLoader(subject_id=3, config=cfg, is_train=True)

# build model
step_input = mx.symbol.Variable('in_features')
initial_states = mx.symbol.Variable('initial_state')


rnn_type = cfg['net']['rnn_type']
num_hidden = cfg['net']['num_hidden']
lstm_cell = mx.rnn.LSTMCell(num_hidden=num_hidden)

output, states = lstm_cell()