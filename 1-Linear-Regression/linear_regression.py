# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 22:08:00 2020

@author: Ali Onar
"""

#import library
# Dataseti okumak içinde işlem yapılabilir hale getirmek için pandas kütüphanesini çağırdık
import pandas as pd
#oluşan modeli görselleştirmek için matplotlib kütüphanesi çağırdık
import matplotlib.pyplot as plt

#import data
df = pd.read_csv("linear_regression_dataset.csv", sep=";")

#grafiksel görsel için plt.scatter
plt.scatter(df.Deneyim, df.Maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show

#%% linear regression or line fit

# sklearn library
from sklearn.linear_model import LinearRegression

# linear regression modelini linear_reg değişkenine eşitledik
linear_reg = LinearRegression()

#deneyim verileri
x = df.Deneyim.values.reshape(-1,1)
#maas verileri
y = df.Maas.values.reshape(-1,1)

#verdiğimiz verileri, tahmin edilen verilere uygulayalım
linear_reg.fit(x,y)

#%% prediction

import numpy as np

# x = 0 olduğunda y'de kesişen noktanın değeri
# ikiside aynı sonucu verir.
b0 = linear_reg.predict([[0]])
print("b0: ",b0)

b0_ = linear_reg.intercept_ # y eksenini kestiği nokta
print("b0_: ",b0_)

b1 = linear_reg.coef_ #eğim
print("b1: ", b1)

"""
artık b0 ve b1 yani y eksenini kesen değer ile eğimi biliyoruz
maas = b0 + b1 * deneyim 
işlemini yapabiliriz
"""
maas1 = -672 + 1270 * 9
print("Maaş1: ", maas1)

"""
ya da aşağıdaki gibi kullanabiliriz
print(linear_reg.predict([[9]]))
"""

#visualize line

# x eksenimiz yani deneyim'i oluşturalım
array = np.array ([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]).reshape(-1,1)

#tahmin edilen değer yani maas'ı oluşturalım
y_head = linear_reg.predict(array)

#çizgiyi göster
plt.plot(array, y_head, color = "red")

#verileri göster
plt.scatter(x,y, color="blue")
plt.show()













