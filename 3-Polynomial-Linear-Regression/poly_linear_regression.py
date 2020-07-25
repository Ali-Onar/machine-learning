# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 20:45:54 2020

@author: Ali Onar
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial_linear_regression_dataset.csv", sep=";")

"""
df.yas diyerek veri setinden bir seri oluşturduk
values ekleyerek array'e çevirdik
reshape(-1,1) yaparakta diziyi biraz sonra kullanacağımız Sklearn kütüphanesinin
okuyabileceği şekle çevirdik
"""
x = df.yas.values.reshape(-1,1)
y = df.boy.values.reshape(-1,1)



#%% linear regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x,y)

#%% predict

y_head = lr.predict(x)

plt.plot(x, y_head, color="red", label="Linear")
plt.scatter(x,y)
plt.xlabel("Yas")
plt.ylabel("Boy")
plt.legend()
plt.show()

lr.predict([[55]])

#%% polynomial regression = y = b0 + b1*x + b2*x^2 + b3*x^3 +bn*x^n

from sklearn.preprocessing import PolynomialFeatures
# degree, n demektir
poly_reg = PolynomialFeatures(degree = 4)

x_polynomial = poly_reg.fit_transform(x)
"""
fit.transform işlemi x değerimizin yani yas'taki tüm verilerin karesini aldırır
"""

#%% fit

lr2 = LinearRegression()
lr2.fit(x_polynomial, y)


#%% visualize

y_head2 = lr2.predict(x_polynomial)

plt.plot(x, y_head2, color="green", label="Polynomial")
plt.scatter(x,y)
plt.xlabel("Yas")
plt.ylabel("Boy")
plt.legend()
plt.show()
