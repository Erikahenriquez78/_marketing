import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import pickle
import os
from imblearn.over_sampling import SMOTE


# Obtener la ruta del directorio actual
directorio_actual = os.getcwd()

# Construir la ruta completa al archivo train.csv en la carpeta "data"
ruta_train = os.path.join(directorio_actual, "data", "train.csv")

# Leer el archivo CSV en un DataFrame
data = pd.read_csv(ruta_train)



x = ['Year_Birth', 'Income', 'Recency', 'MntWines',
            'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
            ]  # Variables predictoras
y = 'Response'  # Variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data[x], data[y], test_size=0.2, random_state=42)





# Aplicar SMOTE para oversampling
oversampler = SMOTE(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(data[x], data[y])

# Crear un nuevo DataFrame con las características y el objetivo balanceados
balanced_data = pd.DataFrame(X_resampled, columns=x)
balanced_data[y] = y_resampled


# Crear el pipeline con escalado de características y modelo de clasificación
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', None)
    
    ])

param_grid =[{
        'classifier': [DecisionTreeClassifier()],
        'classifier__max_depth': [8, 10, 15]
    },
        
    {'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [50, 100, 200]}]
    
              
              
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Obtener los resultados de la búsqueda de hiperparámetros
results = grid_search.cv_results_
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_


# Evaluar el modelo en el conjunto de prueba
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Imprimir las métricas de evaluación
print("Métricas de evaluación en el conjunto de prueba:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC AUC:", roc_auc)



