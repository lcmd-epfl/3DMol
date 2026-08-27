#!/usr/bin/env python3

import pandas as pd

for path in ['data_raw.csv', 'data.csv']:
    print(path)
    df = pd.read_csv(path, dtype={'idx': str})
    print(sum(df['rot589_sign']!=df['rot633_sign']))
    print(sum(df['rot589_sign']!=df['rot355_sign']))
    print()

