import pickle
# from pathlib import Path

# def read_pickle(file_path: str):
#     """
#     Read data from a Pickle file.

#     Args:
#         file_path (str): Path of the Pickle file.

#     Returns:
#         object: Data loaded from the Pickle file, or None if there is an error.
#     """
#     try:
#         with open(file_path, "rb") as f:
#             data = pickle.load(f)
#         print("Pickle file read: OK")
#         return data
#     except (FileNotFoundError, IOError, pickle.PickleError) as err:
#         print(f"Failed to read Pickle file {file_path}: {err}")
#         return None

# Leer el modelo desde el archivo Pickle

# import joblib

# Cargar el modelo desde el archivo .pkl
# loaded_model = joblib.load('_marketing\models\mejor_modelo.pkl')


# modelo = read_pickle("models\mejor_modelo.pkl")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from model import best_model
import os
import pandas as pd

# Directorio actual
directorio_actual = os.getcwd()

# Ruta del archivo CSV de test
ruta_archivo = os.path.join(directorio_actual, 'data',"test.csv")

# Leer el archivo CSV
test = pd.read_csv(ruta_archivo)

# import os
# import pandas as pd
# import pickle
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# # Cargar los datos de prueba
# test_data = pd.read_csv('../data/test.csv')

# Seleccionar las características
# features = ['Year_Birth', 'Income', 'Recency', 'MntWines',
#             'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
#             'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
#             'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
#             'Z_Revenue']  # Variables predictoras

# # Cargar el modelo guardado
# ruta_modelo = os.path.join(os.getcwd(), 'models', 'mejor_modelo1.pkl')
# with open(ruta_modelo, 'rb') as file:
#     best_model = pickle.load(file)

# # Preprocesar los datos de prueba
# X_test = test[features]

# # Realizar las predicciones
# y_pred = best_model.predict(X_test)

# # Imprimir las predicciones
# print("Predicciones:")
# print(y_pred)




# # test = pd.read_csv('data\\test.csv',sep='\t')
# # median_income = test['Income'].median()
# # test['Income'].fillna(median_income, inplace=True)

# features = ['Year_Birth', 'Income', 'Recency', 'MntWines',
#             'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
#             'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
#             'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
#             'Z_Revenue'] 
# target = 'Response' 

# X = test[features]
# y = test[target]
# # predicciones
# predictions = modelo.predict(X)



# from sklearn.metrics import roc_auc_score
# accuracy = accuracy_score(y, predictions)
# precision = precision_score(y, predictions)
# recall = recall_score(y, predictions)
# f1 = f1_score(y, predictions)
# roc_auc = roc_auc_score(y, predictions)


# # Imprimir las predicciones
# print(predictions)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle


features = ['Year_Birth', 'Income', 'Recency', 'MntWines',
            'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
            'Z_Revenue']  # Variables predictoras
target = 'Response'  # Variable objetivo
from sklearn.model_selection import train_test_split
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Aplicar SMOTE para oversampling
oversampler = SMOTE(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(data[features], data[target])

# Crear un nuevo DataFrame con las características y el objetivo balanceados
balanced_data = pd.DataFrame(X_resampled, columns=features)
balanced_data[target] = y_resampled

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(balanced_data[features], balanced_data[target], test_size=0.2, random_state=42)


# Crear el pipeline con escalado de características y modelo de clasificación
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', None)
])
# Obtener la ruta completa al archivo pickle en el directorio actual
ruta_modelo = os.path.join(os.getcwd(), 'models','mejor_modelo1.pkl')

# Abrir el archivo pickle y cargar el modelo
with open(ruta_modelo, 'rb') as file:
    best_model = pickle.load(file)
predictions = modelo.predict(X)
# Evaluar el modelo en el conjunto de prueba
y_pred = best_model.predict(X_test)

