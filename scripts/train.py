import os
import yaml
import pprint
from datetime import datetime
import mxnet as mx

from cgm.loader import CgmLoader
from cgm.symbol import train_cgm_lstm
from cgm.utils.params import summary
from cgm.utils.log import init_logger
from cgm.utils.dir import mkdir_if_not_exist

with open('./config/cgm_lstm.yaml') as f:
    cfg = yaml.load(f)

subject_id = cfg['training']['subject_id']

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

train_data = CgmLoader(subject_id=subject_id, config=cfg, batch_size=batch_size, is_train=True)
val_data = CgmLoader(subject_id=subject_id, config=cfg, batch_size=batch_size, is_train=False)

# build model
sym = train_cgm_lstm(rnn_num_hidden=cfg['net']['num_hidden'], time_steps=cfg['data']['time_steps'])

# summary
info = summary(sym, initial_state=data_shapes[0][1], in_features=data_shapes[1][1], ground_truth=label_shapes[0][1])
print(pprint.pformat(info, indent=2))

model = mx.mod.Module(sym, data_names=data_names, label_names=label_names, context=mx.cpu())

init = mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)
lr_scheduler = mx.lr_scheduler.FactorScheduler(step=1000, factor=0.95)

optimizer = 'adam'
optimizer_params = {'learning_rate': float(cfg['training']['optim']['learning_rate']),
                    'wd': float(cfg['training']['optim']['weight_decay']),
                    'clip_gradient': float(cfg['training']['optim']['clip_gradient']),
                    'rescale_grad': 1.0 / batch_size,
                    'lr_scheduler': lr_scheduler}

metric = mx.metric.Loss(output_names=["l2_loss_output"], name="l2_loss")

# initialize logger
log_root_dir = cfg['training']['logdir']
prefix = '{}_pat{}'.format(cfg['net']['name'], subject_id)
tag = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
logdir = os.path.join(log_root_dir, prefix, tag)
mkdir_if_not_exist(logdir)
logger = init_logger(log_root_dir, prefix, tag)
logger.info(pprint.pformat(cfg, indent=2))

model.fit(train_data=train_data,
          eval_data=val_data,
          eval_metric=metric,
          validation_metric=metric,
          initializer=init,
          optimizer=optimizer,
          optimizer_params=optimizer_params,
          num_epoch=cfg['training']['num_epoch'],
          allow_missing=True,
          batch_end_callback=mx.callback.Speedometer(frequent=10, batch_size=batch_size),
          epoch_end_callback=mx.callback.do_checkpoint(os.path.join(logdir, tag), period=30),
          kvstore='local')

