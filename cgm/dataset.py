from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .utils import parse_single_pat_files, parse_data_files, read_one_subject_data, select_time_window
from .preprocess import butter_lowpass_filter


class SinglePatDataset(Dataset):
    def __init__(self, data_dir, subject_id, begin_date, end_date, in_features, transform=None):
        self.root_dir = data_dir
        self.subject_id = subject_id
        self.begin_date = begin_date
        self.end_date = end_date
        self.in_features = in_features
        self.data = self._load_cgm_data()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

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
        x = np.arange(s, e, 5.) # timestamp
        y = np.zeros((x.shape[0], k)) # input features

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
                vt = all_data[k]['minutes'].values
                v = all_data[k][k].values
                tv = np.interp(x, vt, v)
                tv = butter_lowpass_filter(tv, cutoff_frequency, sample_rate / 2)
                y[:, i] = tv

        in_seqs = y[:-1, :]

        # cgm change rate
        y = np.zeros((x.shape[0], self.num_out))
        for i, k in enumerate(self.out_feat):
            vt = all_data[k]['minutes'].values
            v = all_data[k][k].values
            tv = np.interp(x, vt, v)
            tv = butter_lowpass_filter(tv, cutoff_frequency, sample_rate / 2)
            y[:, i] = tv
        bgv = np.zeros((x.shape[0]-1, self.num_out))
        bgv[:, 0] = np.diff(y[:, 0])

        # normalization
        ynorm = np.zeros_like(in_seqs)
        for i in range(ynorm.shape[1]):
            ynorm[:, i] = (in_seqs[:, i] - in_seqs[:, i].min()) / (in_seqs[:, i].max() - in_seqs[:, i].min())
        in_seqs = ynorm

        ynorm = np.zeros_like(bgv)
        for i in range(ynorm.shape[1]):
            ynorm[:, i] = 0.01 * (bgv[:, i] - np.mean(bgv[:, i]))

        # Generate sequences
        state_size = self.hist_len
        time_steps = self.seq_len

        in_data = []
        out_data = []
        hist_data = []
        all_data = []

        t1 = state_size
        t2 = t1 + time_steps
        while t2 < bgv.shape[0]:
            in_data.append(in_seqs[t1:t2])
            out_data.append(bgv[t1:t2])
            hist_data.append(bgv[t1-state_size:t1])
            all_data.append([in_seqs[t1:t2], bgv[t1:t2], bgv[t1-state_size:t1]])
            t1 += 2
            t2 = t1 + time_steps

        in_data = np.array(in_data)
        out_data = np.array(out_data)
        hist_data = np.array(hist_data)

        num_train = int(len(all_data) * 0.9)
        if self.phase == "TRAIN":
            all_data = all_data[:num_train]
        else:
            all_data = all_data[num_train:]

        print("Input sequence shape: {}".format(in_data.shape))
        print("Output sequence shape: {}".format(out_data.shape))
        print("History sequence shape: {}".format(hist_data.shape))
        print("#Sample: {}".format(len(all_data)))

        return all_data


class AllPatDataset(Dataset):
    def __init__(self, cgm_csv_files, mi_csv_files, transform=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
