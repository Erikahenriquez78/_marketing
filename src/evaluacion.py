import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from model import best_model
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Ruta del archivo CSV de test
# Directorio actual
directorio_actual = os.getcwd()

# Ruta del archivo CSV de test
ruta_archivo = os.path.join(directorio_actual, 'data',"test.csv")

# Leer el archivo CSV
test = pd.read_csv(ruta_archivo)

# Obtener las características utilizadas durante el entrenamiento del modelo
features = ['Year_Birth', 'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
            'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

# Filtrar las características en los datos de prueba
test_features = test[features]


# Cargar el modelo previamente entrenado desde el archivo pkl
ruta_modelo = os.path.join(os.getcwd(), 'models', 'modelo1.pkl')
with open(ruta_modelo, 'rb') as file:
    best_model = pickle.load(file)


y_pred = best_model.predict(test_features)
y_true = test['Response'].astype(int)
# Imprimir las predicciones
print(y_pred)


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

# Imprimir las métricas de evaluación
print("Métricas de evaluación en el conjunto de TEST:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC AUC:", roc_auc)
