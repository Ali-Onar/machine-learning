import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Decision Tree'de kullandığımız veri setini kullanıyoruz
df = pd.read_csv("decision_tree_regression_dataset.csv",sep = ";")

# x (seviye) ve y (ücret) eksenlerimizi oluşturalım

# 0'ıncı indexteki tüm satırları al
# 1'inci indexteki tüm satırları al
x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)
# values ile numpy'a çevirdik reshape ile sklearn'ün anlayabileceği diziye çevirdik

# %% Random Forest Regression

from sklearn.ensemble import RandomForestRegressor
# Neden ensemble? 
# Çünkü Random Forest, Ensemble Learning'in Makine Öğrenimi algoritmalarındandır

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
"""
n_estimators: kaç tane tree kullanacağımızı söylüyoruz

random_state: Elimizdeki verinin bir kısmını eğitim verisine (%70) bir kısmınıda test verisine (%30) 
böldüğmüzü varsayalım. Programı bu şekilde yazdığımızda veriyi her seferinde farklı yerlerden bölmüş olacağız. 
random_state değeri belirlediğimizde ise veriyi her seferinde aynı yerinden bölmüş oluyoruz yani aynı test verileriyle
test etmiş oluyoruz.
"""
rf.fit(x,y)

print("9.5 seviyesinde ücret ne kadar: ",rf.predict([[9.5]]))

# x_grid'nin min değerinden max değerine 0.01'lik sayılarla git
x_grid = np.arange(min(x),max(x),0.01).reshape(-1,1)
# her bir x_grid değeri için y_head hesapla
y_head = rf.predict(x_grid)

#%% görselleştirme
plt.scatter(x,y,color="red")

# x_grid : tahmin etmek istediğim değerler, y_head : tahmin sonuçlarım
plt.plot(x_grid, y_head, color="green")
plt.xlabel("seviye")
plt.ylabel("ucret")
plt.show()


"""
Sonuç: Decision Tree'deki gibi aynı sonuç geldi farkları burada yüz tane Decision Tree kullanıldı.
Buradan şunu söyleyebiliriz ki Random Forest Algoritması Decision Tree Algoritmasından
daha iyi sonuçlar vermektedir.
"""




