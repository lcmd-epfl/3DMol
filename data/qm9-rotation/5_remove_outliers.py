import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('data.csv', dtype={'idx': str}, index_col='idx')

df.drop(labels='023364', inplace=True)    # first excited state at 591.15nm
df.drop(labels='071536', inplace=True)    # first excited state at 630.55nm

def find_outliers(df, threshold):
    k, b, r_value, _, _ = stats.linregress(df['x'], df['y'])
    df['y_pred'] = b + k * df['x']
    df['dy'] = df['y'] - df['y_pred']
    df['outlier'] = np.abs(df['dy']) > threshold * df['dy'].std()
    print(f"{k= :4.2f}\t{b= :6.2f}\tR^2={np.sign(r_value)*r_value**2 :5.2f}\toutliers= {sum(df['outlier']):2}")

def plot_outliers():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.scatterplot(x='x', y='y', data=df, hue='outlier', palette={False: 'blue', True: 'red'})
    plt.plot(df['x'], df['y_pred'], color='green')
    plt.show()

df['x'] = df['rot589']

# for 589 and 633
df['y'] = df['rot633']
df['outlier'] = False
for threshold in (20, 20, 40, 60):
    df.drop(df[df['outlier']].index, inplace=True)
    find_outliers(df, threshold)

print()
# for 355
df['y'] = df['rot355']
df['outlier'] = False
threshold = 20
for i in range(12):
    df.drop(df[df['outlier']].index, inplace=True)
    find_outliers(df, threshold)

df.drop(labels=['x', 'y', 'y_pred', 'dy', 'outlier'], axis=1, inplace=True)
df.to_csv('data.csv')
