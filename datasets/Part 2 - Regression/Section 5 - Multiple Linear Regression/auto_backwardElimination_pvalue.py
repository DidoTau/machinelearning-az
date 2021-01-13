
import statsmodels.api as sm
import numpy as np
import pandas as pd

# Importar el data set
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) #la comuna de estados, pasa a numeros
onehotencoder = make_column_transformer((OneHotEncoder(), [3]), remainder = "passthrough") #pasa a variables dummy, pone las dymmy en primer lugar
X = onehotencoder.fit_transform(X)

# Evitar la trampa de las variables ficticias
X = X[:, 1:] # elimino una para evitar multicolinealidad

# Agrega una columna de 1 por el término cte del modelo
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
def backwardElimination(x, sl) :
    numVars = len(x[0])
    for i in range(0, numVars) :
        regressor_OLS = sm.OLS(y, x.tolist()).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl :
            for j in range(0, numVars - i) :
                if (regressor_OLS.pvalues[j].astype(float) == maxVar) :
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x


SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)