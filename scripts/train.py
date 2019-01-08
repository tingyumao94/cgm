import yaml
import pprint
import mxnet as mx

from cgm.loader import CgmLoader
from cgm.utils.params import summary

with open('./config/train_cgm_lstm.yaml') as f:
    cfg = yaml.load(f)

# load data
batch_size = cfg['training']['batch_size']
time_steps = cfg['data']['time_steps']
hist_length = cfg['data']['hist_length']
num_in = len(cfg['data']['in_features'])
num_out = 1

data_names = ['initial_state', 'in_features']
data_shapes = [('initial_state', (batch_size, hist_length)), ('in_features', (batch_size, time_steps, num_in))]
label_names = ['ground_truth']
label_shapes = [('ground_truth', (batch_size, time_steps, 1))]

train_data = CgmLoader(subject_id=3, config=cfg, is_train=True)
val_data = CgmLoader(subject_id=3, config=cfg, is_train=True)

# build model
in_data = mx.symbol.Variable('in_features')
initial_states = mx.symbol.Variable('initial_state')
gt_data = mx.symbol.Variable('ground_truth')

rnn_type = cfg['net']['rnn_type'] # TODO: chooose rnn_type, like LSTM or GRU
num_hidden = cfg['net']['num_hidden']
lstm_cell = mx.rnn.LSTMCell(num_hidden=num_hidden)

# latent state encoder
h_begin_states = mx.symbol.FullyConnected(data=initial_states, num_hidden=num_hidden, name='initial_h_encoder')
c_begin_states = mx.symbol.FullyConnected(data=initial_states, num_hidden=num_hidden, name='initial_c_encoder')

# rnn
outputs, states = lstm_cell.unroll(length=time_steps, inputs=in_data,
                                   begin_state=(h_begin_states, c_begin_states), merge_outputs=True, layout='NTC')

# output decoder
prediction = mx.symbol.FullyConnected(data=outputs, num_hidden=1, flatten=False, name='cgm_decoder')

# loss
l2_loss = mx.symbol.MakeLoss(0.5 * (prediction - gt_data) ** 2, name='l2_loss')

# # visualization
# graph = mx.viz.plot_network(l2_loss, shape={k: v for k, v in data_shapes + label_shapes}, save_format='pdf')
# graph.render('./samples/reid_train_net', view=True)

# summary
info = summary(l2_loss, initial_state=data_shapes[0][1], in_features=data_shapes[1][1], ground_truth=label_shapes[0][1])
print(pprint.pformat(info, indent=2))

# model = mx.mod.Module(l2_loss, data_names=data_names, label_names=label_names, context=mx.cpu())
init = mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)
# lr_scheduler = mx.lr_scheduler.FactorScheduler(step=1000, factor=0.95)
#
# optimizer = 'adam'
# optimizer_params = {'learning_rate': cfg['training']['optim']['learning_rate'],
#                     'wd': cfg['training']['optim']['weight_decay'],
#                     'clip_gradient': cfg['training']['optim']['clip_gradient'],
#                     'rescale_grad': 1.0 / batch_size,
#                     'lr_scheduler': lr_scheduler}
#
# metric = mx.metric.Loss(output_names=["l2_loss_output"], name="l2_loss")

model = mx.mod.Module(prediction, data_names=data_names, label_names=label_names, context=mx.cpu())
model.bind(for_training=False, data_shapes=data_shapes, force_rebind=True)
model.init_params(initializer=init)

# train model
# model.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label, for_training=True)
# model.init_params(initializer=init)
# model.init_optimizer(kvstore='local', optimizer=optimizer, optimizer_params=optimizer_params)
#
# data_batch = next(train_data)
# model.forward(data_batch, is_train=False)

model.fit(train_data=train_data,
          eval_data=val_data,
          eval_metric=metric,
          validation_metric=metric,
          initializer=init,
          optimizer=optimizer,
          optimizer_params=optimizer_params,
          num_epoch=10,
          allow_missing=True,
          kvstore='local')

