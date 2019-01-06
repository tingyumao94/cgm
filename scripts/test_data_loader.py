import yaml
from cgm.loader import CgmLoader


with open('config/train_cgm_lstm.yaml') as f:
    cfg = yaml.load(f)

test_data = CgmLoader(subject_id=3, config=cfg, is_train=False)

for data_batch in test_data:
    print(data_batch)