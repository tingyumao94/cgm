import os
import yaml
import pprint
import numpy as np
import mxnet as mx

from cgm.loader import CgmLoader
from cgm.symbol import test_cgm_lstm
from cgm.utils.params import summary
from cgm.utils.log import init_logger
from cgm.utils.dir import mkdir_if_not_exist
from cgm.utils.params import load_checkpoint

with open('./config/train_cgm_lstm.yaml') as f:
    cfg = yaml.load(f)

subject_id = cfg['testing']['subject_id']

# load data
batch_size = cfg['test']['batch_size']
time_steps = cfg['data']['time_steps']
hist_length = cfg['data']['hist_length']
num_in = len(cfg['data']['in_features'])
num_out = 1

data_names = ['initial_state', 'in_features']
data_shapes = [('initial_state', (batch_size, hist_length)), ('in_features', (batch_size, time_steps, num_in))]
label_names = ['ground_truth']
label_shapes = [('ground_truth', (batch_size, time_steps, 1))]

val_data = CgmLoader(subject_id=subject_id, config=cfg, is_train=False)

# build model
sym = test_cgm_lstm(rnn_num_hidden=cfg['net']['num_hidden'], time_steps=cfg['data']['time_steps'])

# summary
info = summary(sym, initial_state=data_shapes[0][1], in_features=data_shapes[1][1], ground_truth=label_shapes[0][1])
print(pprint.pformat(info, indent=2))

model = mx.mod.Module(sym, data_names=data_names, label_names=label_names, context=mx.cpu())
model.bind(for_training=False, data_shapes=data_shapes, label_shapes=label_shapes, force_rebind=True)

# load model
model_load_prefix = cfg['testing']['model_load_prefix']
model_load_epoch = cfg['testing']['model_load_epoch']
arg_params, aux_params = load_checkpoint(model_load_prefix, model_load_epoch)
model.set_params(arg_params=arg_params, aux_params=aux_params, allow_extra=True)

all_predictions = []
all_gt = []
for data_batch in val_data:
    gt = data_batch.label[0].asnumpy()
    model.forward(data_batch=data_batch, is_train=False)
    predictions, l2_loss = model.get_outputs()
    predictions = predictions.asnumpy()
    all_predictions.append(predictions)
    all_gt.append(gt)

# prediction
time_points = [3, 6, 12]
pred_err_15 = []
pred_err_30 = []
pred_err_60 = []
for pred, gt in zip(all_predictions, all_gt):
    pred_cumsum = np.cumsum(pred, axis=1)
    gt_cumsum = np.cumsum(gt, axis=1)
    pred_err_15.append()


