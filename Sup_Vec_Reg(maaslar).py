# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# veri yükleme
veriler = pd.read_csv("maaslar.csv")

#data frame dilimleme (slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#dilimleri numpy array(dizi) e çevirdik
X = x.values
Y = y.values

# Linear regression 
# doğrusal model oluşturma
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)

# görselleştirme
plt.scatter(X,Y)
plt.plot(X,lin_reg.predict(X))
plt.show()

#Polynomial regression
#doğrusal olmayan (nonlinear) model oluşturma
from sklearn.preprocessing import PolynomialFeatures

# 2. dereceden polinom
# burada önceki X değerini poly_reg ile derecelerini alıp dönüştürüyoruz.(X0,X1,X2)
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)

#görselleştirme
plt.scatter(X,Y)
plt.plot(X,lin_reg2.predict(x_poly))
plt.show()

# 3. dereceden polinom
poly_reg = PolynomialFeatures(degree = 3)
x_poly = poly_reg.fit_transform(X)
print(x_poly)

lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly,Y)

#görselleştirme
plt.scatter(X,Y)
plt.plot(X,lin_reg3.predict(x_poly))
plt.show()


#↨ tahminler

# Linear Regression
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

# Polynom Regression (3rd degree)
print(lin_reg3.predict(poly_reg.fit_transform([[11]])))
print(lin_reg3.predict(poly_reg.fit_transform([[6.6]])))



#verilerin olceklenmesi (SVR da önemlidir)
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()
y_olcekli = sc2.fit_transform(Y)



from sklearn.svm import SVR

svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli)
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),"r")

# Predict (SVR)

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

















