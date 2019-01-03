import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_single_pat_files(data_dir, subject_id):

    subject = "subject{}".format(subject_id)
    # parse files
    sub_dict = dict()
    sub_dir = os.path.join(data_dir, subject)
    if not os.path.isdir(sub_dir):
        os.makedirs(sub_dir)
    sub_files = os.listdir(sub_dir)
    for f in sorted(sub_files):
        if "activity" in f or "Activity" in f:
            sub_dict['activity'] = os.path.join(sub_dir, f)
        elif "_CGM_" in f or "Glucose" in f:
            sub_dict['CGM'] = os.path.join(sub_dir, f)
        elif "_heartrate_" in f or "HR" in f:
            sub_dict['hr'] = os.path.join(sub_dir, f)
        elif "_sleep timeseries_" in f or "Sleep Timeseries" in f:
            sub_dict['sleep_ts'] = os.path.join(sub_dir, f)
        elif "_sleep_" in f or "Sleep" in f:
            sub_dict['sleep'] = os.path.join(sub_dir, f)
        elif "_summary_" in f:
            sub_dict['summary'] = os.path.join(sub_dir, f)
        elif "food" in f and f.endswith(".csv"):
            sub_dict['food'] = os.path.join(sub_dir, f)
        elif "insulin" in f and f.endswith(".csv"):
            sub_dict['insulin'] = os.path.join(sub_dir, f)

    return sub_dict

def parse_data_files(data_dir, num_subject):
    # parse files
    file_dict = dict()
    subjects = ["subject{}".format(i) for i in range(1, num_subject + 1)]
    for s in subjects:
        sub_dict = dict()
        sub_dir = os.path.join(data_dir, s)
        if not os.path.isdir(sub_dir):
            os.makedirs(sub_dir)
        sub_files = os.listdir(sub_dir)
        for f in sorted(sub_files):
            if "activity" in f or "Activity" in f:
                sub_dict['activity'] = os.path.join(sub_dir, f)
            elif "_CGM_" in f or "Glucose" in f:
                sub_dict['CGM'] = os.path.join(sub_dir, f)
            elif "_heartrate_" in f or "HR" in f:
                sub_dict['hr'] = os.path.join(sub_dir, f)
            elif "_sleep timeseries_" in f or "Sleep Timeseries" in f:
                sub_dict['sleep_ts'] = os.path.join(sub_dir, f)
            elif "_sleep_" in f or "Sleep" in f:
                sub_dict['sleep'] = os.path.join(sub_dir, f)
            elif "_summary_" in f:
                sub_dict['summary'] = os.path.join(sub_dir, f)
            elif "food" in f and f.endswith(".csv"):
                sub_dict['food'] = os.path.join(sub_dir, f)
            elif "insulin" in f and f.endswith(".csv"):
                sub_dict['insulin'] = os.path.join(sub_dir, f)
        file_dict[s] = sub_dict

    return file_dict


def read_one_subject_data(sub_files):
    cgm_file = sub_files['CGM']
    hr_file = sub_files['hr']
    cal_file = sub_files['activity']
    sleep_file = sub_files['sleep_ts']

    insulin_file = sub_files['insulin']
    food_file = sub_files['food']

    # read blood glucose data
    cgm_data = pd.read_csv(cgm_file)
    cgm_data = cgm_data.dropna(subset=['Timestamp (YYYY-MM-DDThh:mm:ss)'])
    cgm_data['time'] = pd.to_datetime(cgm_data['Timestamp (YYYY-MM-DDThh:mm:ss)'])
    cgm_data['minutes'] = pd.to_timedelta(cgm_data['time']).dt.total_seconds() / 60.
    # drop invalid rows
    bg_data = cgm_data.dropna(subset=['Glucose Value (mg/dL)'])
    bg_data = bg_data[['time', 'minutes', 'Glucose Value (mg/dL)']]
    bg_data.loc[bg_data['Glucose Value (mg/dL)'].isin(['High']), 'Glucose Value (mg/dL)'] = 400.
    bg_data.loc[bg_data['Glucose Value (mg/dL)'].isin(['Low']), 'Glucose Value (mg/dL)'] = 40.
    bg_data['bg'] = bg_data['Glucose Value (mg/dL)'].astype('float')
    bg_data = bg_data[['time', 'minutes', 'bg']]
    assert min(bg_data['bg']) >= 40 and max(bg_data['bg']) <= 400, "Invalid blood sugar range"

    # read heart rate data
    hr_data = pd.read_csv(hr_file)
    hr_data = hr_data.dropna(subset=['HEART RATE DATE/TIME', 'VALUE'])
    hr_data['time'] = pd.to_datetime(hr_data['HEART RATE DATE/TIME'])
    hr_data['minutes'] = pd.to_timedelta(hr_data['time']).dt.total_seconds() / 60.
    hr_data = hr_data[['time', 'minutes', 'VALUE']]
    hr_data.columns = ['time', 'minutes', 'hr']

    # read activity data
    cal_data = pd.read_csv(cal_file)
    cal_data = cal_data.dropna(subset=['ACTIVITY DATE/TIME', 'CALORIES', 'STEPS', 'DISTANCE', 'FLOORS'])
    cal_data['time'] = pd.to_datetime(cal_data['ACTIVITY DATE/TIME'])
    cal_data['minutes'] = pd.to_timedelta(cal_data['time']).dt.total_seconds() / 60.
    cal_data = cal_data[['time', 'minutes', 'CALORIES', 'STEPS', 'DISTANCE', 'FLOORS']]
    cal_data.columns = ['time', 'minutes', 'calories', 'steps', 'distance', 'floors']

    # read sleep series data
    sleep_data = pd.read_csv(sleep_file)
    sleep_data = sleep_data[['Activity Time', 'State']]
    sleep_data['time'] = pd.to_datetime(sleep_data['Activity Time'])
    sleep_data['minutes'] = pd.to_timedelta(sleep_data['time']).dt.total_seconds() / 60.
    sleep_data = sleep_data[['time', 'minutes', 'State']]
    sleep_data.columns = ['time', 'minutes', 'sleep_state']
    # encode sleep_state to numeric
    sleep_data["sleep_state"] = sleep_data["sleep_state"].astype('category')
    sleep_data["sleep_code"] = sleep_data["sleep_state"].cat.codes

    # read food data
    food_data = pd.read_csv(food_file)
    # food_data = food_data[(food_data['Note'] != "missing") & (food_data['Note'] != "Missing")]
    food_data = food_data[(~food_data['Note'].notnull()) & (food_data['gCarbs'] != 0)]
    food_data = food_data.dropna(subset=['Time'])
    food_data['time'] = pd.to_datetime(food_data['Time'])
    food_data['minutes'] = pd.to_timedelta(food_data['time']).dt.total_seconds() / 60.
    food_data = food_data[['time', 'minutes', 'gCarbs']]

    # read insulin data
    insulin_data = pd.read_csv(insulin_file)
    insulin_data['time'] = pd.to_datetime(insulin_data['Time'])
    insulin_data['minutes'] = pd.to_timedelta(insulin_data['time']).dt.total_seconds() / 60.
    insulin_bg_data = insulin_data[['time', 'minutes', 'Blood Glucose']]
    insulin_bg_data = insulin_bg_data.dropna(subset=['Blood Glucose'])
    if insulin_bg_data['Blood Glucose'].dtype == 'object':
        insulin_bg_data = insulin_bg_data[(insulin_bg_data['Blood Glucose'] != 'high')
                                          & (insulin_bg_data['Blood Glucose'] != 'low')]
    insulin_bg_data['Blood Glucose'] = insulin_bg_data['Blood Glucose'].astype('float')

    carb_data = insulin_data[['time', 'minutes', 'Carbohydrate']]
    carb_data = carb_data.dropna(subset=['Carbohydrate'])
    meal_bolus_data = insulin_data[['time', 'minutes', 'Meal Bolus']]
    meal_bolus_data = meal_bolus_data.dropna(subset=['Meal Bolus'])
    correction_bolus_data = insulin_data[['time', 'minutes', 'Correction Bolus']]
    correction_bolus_data = correction_bolus_data.dropna(subset=['Correction Bolus'])

    # store all data into a dictionary
    all_data = dict()
    all_data['bg'] = bg_data
    all_data['hr'] = hr_data
    all_data['sleep_code'] = sleep_data
    all_data['calories'] = cal_data[['time', 'minutes', 'calories']]
    all_data['steps'] = cal_data[['time', 'minutes', 'steps']]
    all_data['distance'] = cal_data[['time', 'minutes', 'distance']]
    all_data['floors'] = cal_data[['time', 'minutes', 'floors']]
    all_data['gCarbs'] = food_data
    all_data['Carbohydrate'] = carb_data
    all_data['Blood Glucose'] = insulin_bg_data
    all_data['Meal Bolus'] = meal_bolus_data
    all_data['Correction Bolus'] = correction_bolus_data

    return all_data


def plot_data(data, start_date, end_date):
    filter_data = dict()
    for k, v in data.items():
        filter_data[k] = select_time_window(v, start_date, end_date)

    num_corvariates = len(filter_data.keys())

    f, axarr = plt.subplots(num_corvariates, 1, figsize=(20, 3 * num_corvariates))
    for i, k in enumerate(filter_data):
        show_data = filter_data[k]
        axarr[i].plot_date(show_data['time'], show_data[k], markersize=2)
        axarr[i].set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])
        axarr[i].set_title(k)


def select_time_window(data, start_date, end_date):
    filter_data = data[(data['time'] >= pd.to_datetime(start_date)) & (data['time'] <= pd.to_datetime(end_date))]
    return filter_data


def split_ts_data(data, key, win, step, ratio=0.8):
    start_time, end_time = int(min(data['minutes'])), int(max(data['minutes']))

    train_seq = [[s, s + int(ratio * win)] for s in range(start_time, end_time, step) if s + win <= end_time]
    test_seq = [[s + int(ratio * win), s + win] for s in range(start_time, end_time, step) if s + win <= end_time]

    assert len(train_seq) == len(test_seq), "len(train_seq) != len(test_seq)"

    train_list = []
    test_list = []
    for i in range(len(train_seq)):
        train_win = train_seq[i]
        test_win = test_seq[i]

        train_data = data[(data['minutes'] >= train_win[0]) & (data['minutes'] <= train_win[1])][
            ['minutes', key]].as_matrix()
        test_data = data[(data['minutes'] >= test_win[0]) & (data['minutes'] <= test_win[1])][
            ['minutes', key]].as_matrix()

        if len(test_data) < 4 or len(train_data) < 20:
            continue

        train_list += [train_data]
        test_list += [test_data]

    return train_list, test_list


def ts_to_mat(data, intervals, output_dir):
    import scipy.io as sio

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    cnt = 0
    duration = 5 * 60 + 30
    for s, e in intervals:
        start = s - 5 * 60
        while start + duration < e:
            train_data = data[(data['minutes'] >= start) & (data['minutes'] <= start + 5 * 60)]
            test_data = data[(data['minutes'] >= start + 5 * 60) & (data['minutes'] <= start + duration)]

            downsample = 1
            sio.savemat(os.path.join(output_dir, "{}.mat".format(str(cnt).zfill(5))),
                        {'xtrain': train_data['minutes'].values[::downsample],
                         'ytrain': train_data['bg'].values[::downsample],
                         'xtest': test_data['minutes'].values[::downsample],
                         'ytest': test_data['bg'].values[::downsample],
                         'prediction_time': start + duration})
            cnt += 1
            start += 20


if __name__ == '__main__':

    data_dir = '../data/type1'
    num_subject = 6
    file_dict = parse_data_files(data_dir, num_subject)

    # read data
    pat = 'subject5'
    sub_files = file_dict[pat]
    all_data = read_one_subject_data(sub_files)

    plt.figure(figsize=(20, 4))
    plt.plot_date(all_data['bg']['time'], all_data['bg']['bg'], markersize=2, label='BG')
    plt.xlim([all_data['bg']['time'][:1].values[0], all_data['bg']['time'][-1:].values[0]])
    plt.legend(fontsize='x-large', loc=1)
    plt.grid(color='b', linestyle='--', linewidth=1)
    plt.show()
