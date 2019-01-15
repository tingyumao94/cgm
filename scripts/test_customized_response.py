import os
import yaml
import pprint
import numpy as np
import mxnet as mx
import cv2
import matplotlib.pyplot as plt

from cgm.loader import CgmLoader
from cgm.symbol import test_cgm_lstm
from cgm.utils.params import summary
from cgm.utils.log import init_logger
from cgm.utils.dir import mkdir_if_not_exist
from cgm.utils.params import load_checkpoint

with open('./config/cgm_lstm.yaml') as f:
    cfg = yaml.load(f)

subject_id = cfg['visualization']['subject_id']

# load custommized treatments
max_treatments = cfg['visualization']['max_treatments']
treatments = cfg['visualization']['treatments']
treatment_names = cfg['visualization']['treatment_names']
treatment_units = cfg['visualization']['treatment_units']
num_markers = len(treatments)

# load data
batch_size = cfg['visualization']['batch_size']
time_steps = cfg['data']['time_steps']
hist_length = cfg['data']['hist_length']
num_in = len(cfg['data']['in_features'])
num_out = 1

data_names = ['initial_state', 'in_features']
data_shapes = [('initial_state', (batch_size, hist_length)), ('in_features', (batch_size, time_steps, num_in))]
label_names = ['ground_truth']
label_shapes = [('ground_truth', (batch_size, time_steps, 1))]

val_data = CgmLoader(subject_id=subject_id, config=cfg, batch_size=batch_size, is_train=False)

# build model
sym = test_cgm_lstm(rnn_num_hidden=cfg['net']['num_hidden'], time_steps=cfg['data']['time_steps'])
model = mx.mod.Module(sym, data_names=data_names, label_names=label_names, context=mx.cpu())
model.bind(for_training=False, data_shapes=data_shapes, label_shapes=label_shapes, force_rebind=True)

# load model
model_load_prefix = cfg['visualization']['model_load_prefix']
model_load_epoch = cfg['visualization']['model_load_epoch']
arg_params, aux_params = load_checkpoint(model_load_prefix, model_load_epoch)
model.set_params(arg_params=arg_params, aux_params=aux_params, allow_extra=True)

# visualization
selected_ind = int(np.random.random() * val_data.size)
fontsize = 10
fig = plt.figure(figsize=(20, 5))
for i in range(num_markers):

    init = val_data.initial_state_data[selected_ind]
    xinput = val_data.in_data[selected_ind]
    out = val_data.out_data[selected_ind]
    bg_prev = val_data.bg_prev_data[selected_ind]
    bg_prev = np.squeeze(bg_prev)
    bg = val_data.bg_data[selected_ind]
    bg = np.squeeze(bg)

    xs = [mx.ndarray.array([x]) for x in [init, xinput]]
    ys = [mx.ndarray.array([y]) for y in [out]]
    data_batch = mx.io.DataBatch(data=xs, label=ys)

    xinput = np.zeros_like(data_batch.data[1].asnumpy())
    # keep original hr, sleep_code etc.
    xinput[:, :, -3:] = data_batch.data[1].asnumpy()[:, :, -3:]
    if i > 0:
        xinput[0, i - 1] = treatments[i] / max_treatments[i]  # np.log(treatments[i]+1)
    data_batch.data[1] = mx.ndarray.array(xinput)

    model.forward(data_batch=data_batch, is_train=False)

    response, l2_loss = model.get_outputs()
    response = response.asnumpy()
    response = np.squeeze(response)
    tmp = np.concatenate([bg_prev, bg_prev[-1] + np.cumsum(response)])
    plt.plot(5 * np.arange(tmp.shape[0]), tmp,
             label="{} = {} {}".format(treatment_names[i], treatments[i], treatment_units[i]))

tmp = np.concatenate([bg_prev, bg])
plt.plot(5 * np.arange(tmp.shape[0]), tmp, label="ground truth")

plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel("Time (min)", fontsize=fontsize)
plt.ylabel("Blood Glucose (mg/dL)", fontsize=fontsize)
plt.legend(prop={'size': fontsize}, loc='lower left', bbox_to_anchor=(1, 0.0))

fig.canvas.draw()
viz = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
viz = viz.reshape(fig.canvas.get_width_height()[::-1] + (3,))
plt.close(fig)
cv2.imshow('iou_dist', viz)
cv2.waitKey(0)
cv2.destroyAllWindows()