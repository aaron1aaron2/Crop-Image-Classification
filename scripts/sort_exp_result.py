import json

import pandas as pd
from pathlib import Path

folder = 'output/2023.1.2_EXP-sample-nums_2022.1.2_Kaggle P100'
target = 'evaluation.json' # regex

result_dt = {}
for path in Path(folder).rglob(target):
    exp_name = path.parent.name
    data = json.load(open(path))

    result_dt[exp_name] = data