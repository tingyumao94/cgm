from cgm.dataset import SinglePatDataset, AllPatDataset
from cgm.net import Net

subject_id = 3
begin_date = '2018-06-25 00:00:00'
end_date = '2018-07-07 00:00:00'
data_dir = './data/type1'
input_features = ["Correction Bolus", "Meal Bolus", "gCarbs", "Carbohydrate", "steps", "sleep_code"]

dataset = SinglePatDataset(data_dir=data_dir, begin_date=begin_date, end_date=end_date, in_features=input_features)
