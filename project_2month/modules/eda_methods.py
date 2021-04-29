import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot(type_plot, title, x, y, data, estimator, color, xlabel, ylabel):
    plt.figure(figsize = (18,10))
    plt.title(title, fontsize=24)
    ax = sns.type_plot(x=x, y=y, data=data, estimator=estimator, color=color)
    ax = ax.set(xlabel=xlabel, ylabel=ylabel)
    plt.xticks(rotation=45)
    return plt.show() 

def merge_table(table_1, table_2, table_col_1, table_col_2, on_col):
    merge_table = pd.merge(table_1,
                    table_2[[table_col_1, table_col_2]],
                    how="inner",
                    on=on_col)
    return merge_table
