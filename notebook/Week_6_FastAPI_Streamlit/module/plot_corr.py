import seaborn as sns
import matplotlib.pyplot as plt

def plot_corr(data):
    fig1,ax = plt.subplots(figsize=(10,10))
    sns.heatmap(abs(data.corr()),annot=True,ax=ax)
    fig2 = sns.pairplot(data,vars = ["magnesium", "flavanoids","nonflavanoid_phenols","proline"], hue = 'alcohol')
    return fig1,fig2