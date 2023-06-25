import pickle
from pathlib import Path

def read_pickle(file_path: str):
    """
    Read data from a Pickle file.

    Args:
        file_path (str): Path of the Pickle file.

    Returns:
        object: Data loaded from the Pickle file, or None if there is an error.
    """
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print("Pickle file read: OK")
        return data
    except (FileNotFoundError, IOError, pickle.PickleError) as err:
        print(f"Failed to read Pickle file {file_path}: {err}")
        return None

# Leer el modelo desde el archivo Pickle

# import joblib

# Cargar el modelo desde el archivo .pkl
# loaded_model = joblib.load('_marketing\models\mejor_modelo.pkl')


modelo = read_pickle("models\mejor_modelo.pkl")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from model import best_model
test = pd.read_csv('data\\test.csv',sep='\t')
median_income = test['Income'].median()
test['Income'].fillna(median_income, inplace=True)

features = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines',
            'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
            'Z_CostContact', 'Z_Revenue'] 
target = 'Response' 

X = test[features]
y = test[target]
# predicciones
predictions = modelo.predict(X)



from sklearn.metrics import roc_auc_score
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)
f1 = f1_score(y, predictions)
roc_auc = roc_auc_score(y, predictions)


# Imprimir las predicciones
print(predictions)

