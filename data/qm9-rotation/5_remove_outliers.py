#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy import stats

csv_path = 'data.csv'
worst = [
        '023364',  # first excited state at 591.15nm
        '071536',  # first excited state at 630.55nm
        ]


def find_outliers(df, threshold):
    k, b, r_value, _, _ = stats.linregress(df['x'], df['y'])
    df['y_pred'] = b + k * df['x']
    df['dy'] = df['y'] - df['y_pred']
    df['outlier'] = np.abs(df['dy']) > threshold * df['dy'].std()
    nout = sum(df['outlier'])
    print(f"{k= :4.2f}\t{b= :6.2f}\tR^2={np.sign(r_value)*r_value**2 :6.3f}\toutliers= {nout:2}\tthreshold={threshold}σ")
    return nout


def plot_outliers():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.scatterplot(x='x', y='y', data=df, hue='outlier', palette={False: 'blue', True: 'red'})
    plt.plot(df['x'], df['y_pred'], color='green')
    plt.show()


def main():
    df = pd.read_csv(csv_path, dtype={'idx': str}, index_col='idx')
    tot = len(df)
    df.drop(labels=worst, axis=0, inplace=True, errors='ignore')

    df['x'] = df['rot589']

    print('589 vs 633')
    df['y'] = df['rot633']
    df['outlier'] = False
    for threshold in (20, 20, 40, 60):
        df.drop(df[df['outlier']].index, inplace=True)
        find_outliers(df, threshold)
    print()

    print('589 vs 355')
    df['y'] = df['rot355']
    df['outlier'] = False
    threshold = 20
    for i in range(12):
        df.drop(df[df['outlier']].index, inplace=True)
        if find_outliers(df, threshold)==0:
            break
    print()

    df.drop(labels=['x', 'y', 'y_pred', 'dy', 'outlier'], axis=1, inplace=True)
    df.to_csv(csv_path)
    print('*** {} total, {} ({:.1f}%) kept, {} ({:.1f}%) removed'.format(tot, kept:=len(df), kept/tot*100, removed:=tot-kept, removed/tot*100))


if __name__=='__main__':
    main()
