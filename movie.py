
import numpy as np
import pandas as pd
import seaborn as sns


import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

df = pd.read_csv('data/OMdb_mojo_clean.csv')

df = df[['IMdb_score', 'Metascore', 'Runtime']]
sns.heatmap(df.corr(), annot =True)
plt.show()

