import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('data.csv', dtype={'idx': str}, index_col='idx')

df.drop(labels='023364', inplace=True)    # first excited state at 591.15nm
df.drop(labels='071536', inplace=True)    # first excited state at 630.55nm

def find_outliers(df, threshold):
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['x'], df['y'])
    df['y_pred'] = intercept + slope * df['x']
    df['dy'] = df['y'] - df['y_pred']
    df['outlier'] = np.abs(df['dy']) > threshold * df['dy'].std()
    print(slope, intercept, r_value, p_value, std_err)
    print(sum(df['outlier']))

def plot_outliers():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.scatterplot(x='x', y='y', data=df, hue='outlier', palette={False: 'blue', True: 'red'})
    plt.plot(df['x'], df['y_pred'], color='green')
    plt.show()

df['x'] = df['rot589']

# for 589 and 633
df['y'] = df['rot633']
find_outliers(df, 20)
df.drop(df[df['outlier']].index, inplace=True)
find_outliers(df, 20)
df.drop(df[df['outlier']].index, inplace=True)
find_outliers(df, 40)
df.drop(df[df['outlier']].index, inplace=True)
find_outliers(df, 60)

# for 355
df['y'] = df['rot355']
df['outlier'] = False
for i in range(16):
    df.drop(df[df['outlier']].index, inplace=True)
    find_outliers(df, 20)

df.drop(labels=['x', 'y', 'y_pred', 'dy', 'outlier'], axis=1, inplace=True)
df.to_csv('data.csv')
