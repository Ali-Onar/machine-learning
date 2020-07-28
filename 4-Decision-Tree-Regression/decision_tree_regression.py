import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("decision_tree_regression_dataset.csv", sep=";") 

# x (seviye) ve y (ücret) eksenlerimizi oluşturalım
x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%% Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor
dec_tree = DecisionTreeRegressor()
dec_tree.fit(x,y)

# predict
# dec_tree.predict([[8,5]])

#%% visualize

"""
grafikte düz bir çizginin oluşmaması için minimum x değeri ve maximum x değerleri arasında 0'lı sayılar ürettik
çünkü herhangi bir leaf'teki tüm x değerlerinin sonucu tek bir değeri vermektedir.
"""
x_ = np.arange(min(x), max(x), 0.01).reshape(-1,1)
y_head = dec_tree.predict(x_)

plt.scatter(x,y, color="red")
plt.plot(x_, y_head, color="blue")
plt.xlabel("Seviye")
plt.ylabel("Ucret")
plt.show()
