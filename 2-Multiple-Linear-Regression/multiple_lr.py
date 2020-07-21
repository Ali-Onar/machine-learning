# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 00:14:04 2020

@author: Ali Onar
"""

import pandas as pd #dataseti okumak için
import numpy as np
import matplotlib.pyplot as plt
#lineer regresyon import ettik, multiple lr ile aralarındaki tek fark özellik (feature) artması
from sklearn.linear_model import LinearRegression

df = pd.read_csv("multiple_linear_regression_dataset.csv", sep=";")


x = df.iloc[:,[1,2]].values 
"""
: kullanarak tüm satırları al dedik
oda_sayisi ve bina_yasi veri setinde 1. ve 2. sütunlarda olduğu için [1,2] kullandık

"""
y = df.konut_fiyati.values.reshape(-1,1) # konut_fiyati y ekseninde

#%%
#modeli değişkene eşitleyip, x ve y'yi kullanarak bize bir line fit et diyoruz
multiple_lr = LinearRegression()
multiple_lr.fit(x,y)

# intercept_ metodu ile b0 (bias) değerimizi buluyoruz
print("b0: ", multiple_lr.intercept_)

# coef_ metodu ile katsayılarımızı buluyoruz
print("b1, b2: ", multiple_lr.coef_)

#oda_sayisi = 2, bina_yasi=2, konut_fiyati ?
konut_fiyati1 = multiple_lr.predict(np.array([[2,2]]))
#oda_sayisi = 4, bina_yasi=2, konut_fiyati ?
konut_fiyati2 = multiple_lr.predict(np.array([[4,2]]))
#oda_sayisi = 6, bina_yasi=2, konut_fiyati ?
konut_fiyati3 = multiple_lr.predict(np.array([[6,2]]))

print("konut_fiyati1:",konut_fiyati1)
print("konut_fiyati2:",konut_fiyati2)
print("konut_fiyati3:",konut_fiyati3)
