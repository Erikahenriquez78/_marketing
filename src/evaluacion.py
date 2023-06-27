import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from model import best_model
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

# Directorio actual
directorio_actual = os.getcwd()

# Ruta del archivo CSV de test
ruta_archivo = os.path.join(directorio_actual, 'data',"test.csv")

# Leer el archivo CSV
test = pd.read_csv(ruta_archivo)


oversampler = SMOTE(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# # Crear un nuevo DataFrame con las características y el objetivo balanceados
# balanced_data = pd.DataFrame(X_resampled, columns=features)
# balanced_data[target] = y_resampled

# Crear el pipeline con escalado de características y modelo de clasificación
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', None)
])



import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle

# features = ['Year_Birth', 'Income', 'Recency', 'MntWines',
#             'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
#             'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
#             'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
#             'Z_Revenue']  # Variables predictoras
# target = 'Response'  # Variable objetivo
# X_test = test[features]
# y_test = test[target]

ruta_modelo = os.path.join(os.getcwd(), 'models', 'mejor_modelo1.pkl')  # Reemplaza 'ruta_del_archivo_del_modelo.pkl' con la ruta correcta
with open(ruta_modelo, 'rb') as file:
    best_model = pickle.load(file)
    
    
# Directorio actual
directorio_actual = os.getcwd()

# Ruta del archivo CSV de test
ruta_archivo = os.path.join(directorio_actual, 'data',"test.csv")

# Leer el archivo CSV
test = pd.read_csv(ruta_archivo)

# Realizar las predicciones en los datos de prueba
y_pred = best_model.predict(test)

# Imprimir las predicciones
print(y_pred)

accuracy = accuracy_score(test, y_pred)
precision = precision_score(test, y_pred)
recall = recall_score(test, y_pred)
f1 = f1_score(test, y_pred)
roc_auc = roc_auc_score(test, y_pred)

# Imprimir las métricas de evaluación
print("Métricas de evaluación en el conjunto de prueba:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC AUC:", roc_auc)


