import os
import json

import pandas as pd
from pathlib import Path


def sort_result(folder:str, output:str=None, target:str='evaluation.json') -> None:
    result_df = pd.DataFrame()
    for path in Path(folder).rglob(target):
        exp_name = path.parent.name
        data = json.load(open(path))

        tmp = pd.DataFrame(data).T.reset_index().rename(columns={'index': 'phase'})
        tmp = pd.DataFrame([exp_name]*len(tmp), columns=['experiment']).join(tmp)

        result_df = pd.concat([result_df, tmp])

    if output == None:
        output = os.path.join(folder, 'result.csv')

    result_df.to_csv(output, index=False)

if __name__ == '__main__':

    # sample-nums experiment
    sort_result('output/2023.1.2_EXP-sample-nums_2022.1.2_Kaggle P100')

    # crop-vs-resize V2
    sort_result('output/2023.1.4_EXP-crop-vs-resize.V2_Kaggle P100')

    # color-mask
    sort_result('output/2023.1.8_EXP-color_mask_Kaggle P100')