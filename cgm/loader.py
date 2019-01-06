from __future__ import print_function, division
import os
import pandas as pd
import numpy as np
import mxnet as mx
import random

from .utils import parse_single_pat_files, parse_data_files, read_one_subject_data, select_time_window
from .preprocess import butter_lowpass_filter


class CgmLoader(mx.io.DataIter):
    def __init__(self, subject_id, config, is_train):

        super(CgmLoader, self).__init__()

        self.is_train = is_train
        self.root_dir = config['data']['root_dir']
        self.split_ratio = config['data']['split_ratio']
        self.in_features = config['data']['in_features']
        self.out_feature = config['data']['out_features']
        self.num_out = 1
        self.batch_size = config['training']['batch_size'] if is_train else config['testing']['batch_size']

        self.time_steps = config['data']['time_steps']
        self.hist_len = config['data']['hist_length']

        self.subject_id = subject_id
        self.begin_date = config['data']['pat{}'.format(subject_id)]['begin_date']
        self.end_date = config['data']['pat{}'.format(subject_id)]['end_date']

        data = self._load_cgm_data()

        self.in_data = data['in_seq']
        self.out_data = data['out_seq']
        self.initial_state_data = data['initial_states']
        self.bg_data = data['bg_data']

        self.data_shapes = [('initial_state', (self.batch_size, self.time_steps)),
                            ('in_features', (self.batch_size, self.time_steps, len(self.in_features)))]
        self.label_shapes = [('gt_bgv', (self.batch_size, self.time_steps, self.num_out))]

        self.size = self.in_data.shape[0]

        self.reset()

    def reset(self):
        self.current = 0

    def next(self):
        if self.iter_next():
            return self.cur_batch
        else:
            raise StopIteration

    def iter_next(self):

        if self.current + self.batch_size > self.size:
            return False
        else:
            xs = [[] for _ in range(len(self.data_shapes))]
            ys = [[] for _ in range(len(self.label_shapes))]

            for bid, ind in enumerate(range(self.current, self.current + self.batch_size)):
                xs[0].append(self.initial_state_data[ind])
                xs[1].append(self.in_data[ind])
                ys[0].append(self.out_data[ind])

            xs = [mx.ndarray.array(x) for x in xs]
            ys = [mx.ndarray.array(y) for y in ys]
            self.cur_batch = mx.io.DataBatch(data=xs, label=ys)

        self.current += self.batch_size
        return True

    @property
    def provide_data(self):
        return self.data_shapes

    @property
    def provide_label(self):
        return self.label_shapes

    def _load_cgm_data(self):
        data_dir = self.root_dir
        subject_id = self.subject_id
        sub_files = parse_single_pat_files(data_dir, subject_id)

        # read data
        all_data = read_one_subject_data(sub_files)

        start_date = self.begin_date
        end_date = self.end_date
        for k in all_data:
            all_data[k] = select_time_window(all_data[k], start_date, end_date)

        s = all_data['bg']['minutes'][:1].values[0]
        e = all_data['bg']['minutes'][-1:].values[0]

        k = len(self.in_features)
        x = np.arange(s, e, 5.)  # timestamp
        y = np.zeros((x.shape[0], k))  # input features

        cutoff_frequency = 5.0
        sample_rate = 200

        for i, k in enumerate(self.in_features):
            if k in ["Correction Bolus", "Meal Bolus", "gCarbs", "Carbohydrate"]:
                tmp = all_data[k]
                tmp = tmp[tmp[k] < 1000]
                if k == "Meal Bolus":
                    tmp = tmp[tmp[k] < 50]

                max_tmp = max(tmp[k])
                tmp = all_data[k]
                tmp[tmp[k] > max_tmp][k] = max_tmp

                cb = tmp[k].values
                cbt = tmp['minutes'].values
                tt = np.minimum(np.round((cbt - s) / 5).astype('int'), x.shape[0] - 1)
                tv = np.zeros_like(x)
                tv[tt] = cb
                y[:, i] = tv
            elif k == "steps":
                stept = all_data['steps']['minutes'].values
                step = all_data['steps']['steps'].values
                tv = np.interp(x, stept, step)
                # why smooth twice here?
                tv = np.convolve(tv, np.ones((12,)) / 12., mode='same')
                tv = butter_lowpass_filter(tv, cutoff_frequency, sample_rate / 2)
                y[:, i] = tv
            elif k == "sleep_code":
                def find_sleep_interval(all_data):
                    sleep_data = all_data['sleep_code']
                    sleep_time_points = sorted(sleep_data['minutes'])
                    sleep_interval = []
                    rest_limit = 60
                    s = sleep_time_points[0]
                    for i, _ in enumerate(sleep_time_points[1:-1]):
                        e = sleep_time_points[i]
                        if sleep_time_points[i + 1] - e > rest_limit:
                            sleep_interval += [[s, e]]
                            s = sleep_time_points[i + 1]
                    return sleep_interval

                sleep_interval = find_sleep_interval(all_data)
                tv = np.zeros_like(x)
                for ss, se in sleep_interval:
                    ss = max(ss, s)
                    se = min(se, e)
                    tv[int((ss - s) // 5):int((se - s) // 5)] = 1.
                y[:, i] = tv
            else:
                # heart rate
                vt = all_data[k]['minutes'].values
                v = all_data[k][k].values
                tv = np.interp(x, vt, v)
                tv = butter_lowpass_filter(tv, cutoff_frequency, sample_rate / 2)
                y[:, i] = tv

        in_seqs = y[:-1, :]

        # cgm change rate
        y = np.zeros((x.shape[0], self.num_out))
        for i, k in enumerate(self.out_feature):
            vt = all_data[k]['minutes'].values
            v = all_data[k][k].values
            tv = np.interp(x, vt, v)
            tv = butter_lowpass_filter(tv, cutoff_frequency, sample_rate / 2)
            y[:, i] = tv
        bg = y
        bgv = np.zeros((x.shape[0] - 1, self.num_out))
        bgv[:, 0] = np.diff(y[:, 0])

        # normalization: input seq
        ynorm = np.zeros_like(in_seqs)
        for i in range(ynorm.shape[1]):
            ynorm[:, i] = (in_seqs[:, i] - in_seqs[:, i].min()) / (in_seqs[:, i].max() - in_seqs[:, i].min())
        in_seqs = ynorm

        # ynorm = np.zeros_like(bgv)
        # for i in range(ynorm.shape[1]):
        #     ynorm[:, i] = 0.01 * (bgv[:, i] - np.mean(bgv[:, i]))

        # Generate sequences
        state_size = self.hist_len
        time_steps = self.time_steps

        all_data = []

        t1 = state_size
        t2 = t1 + time_steps
        while t2 < bgv.shape[0]:
            all_data.append([in_seqs[t1:t2], bgv[t1:t2], bgv[t1 - state_size:t1], bg[t1:t2]])
            t1 += 2
            t2 = t1 + time_steps

        # random shuffle
        random.shuffle(all_data)

        in_data = np.array([x[0] for x in all_data])
        out_data = np.array([x[1] for x in all_data])
        hist_data = np.array([x[2] for x in all_data])
        hist_data = np.squeeze(hist_data)
        bg_data = np.array([x[3] for x in all_data])

        num_train = int(in_data.shape[0] * self.split_ratio)
        if self.is_train == "TRAIN":
            in_data = in_data[:num_train]
            out_data = out_data[:num_train]
            hist_data = hist_data[:num_train]
            bg_data = bg_data[:num_train]
        else:
            in_data = in_data[num_train:]
            out_data = out_data[num_train:]
            hist_data = hist_data[num_train:]
            bg_data = bg_data[num_train:]

        print("Input sequence shape: {}".format(in_data.shape))
        print("Output sequence shape: {}".format(out_data.shape))
        print("History sequence shape: {}".format(hist_data.shape))
        print("Blood glucose sequence shape: {}".format(bg_data.shape))

        return {'bg_data': bg_data, 'in_seq': in_data, 'out_seq': out_data, 'initial_states': hist_data}