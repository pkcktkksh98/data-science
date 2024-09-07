import pandas as pd

def load_data():
    df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',names=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols','flavanoids', 'nonflavanoid_phenols' ,'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315_of_diluted_wines', 'proline'],delimiter=',', index_col=False)
    return df