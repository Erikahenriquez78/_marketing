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

# Cargar los datos
data = pd.read_csv(r'_marketing\data\train.csv',sep='\t')

median_income = data['Income'].median()
data['Income'].fillna(median_income, inplace=True)

# Seleccionar las características y el objetivo
features = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines',
            'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
            'Z_CostContact', 'Z_Revenue']  # Variables predictoras
target = 'Response'  # Variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Aplicar resampling al conjunto de entrenamiento
resampler = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)

# Crear el pipeline con escalado de características y modelo de clasificación
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', None)
])

# Definir los parámetros del grid search para cada modelo
param_grid = [
    {
        'classifier': [LogisticRegression(solver='liblinear')],
        'classifier__C': [0.1, 1, 10]
    },
    {
        'classifier': [DecisionTreeClassifier()],
        'classifier__max_depth': [8, 10, 15]
    },
    {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [50, 100, 200]
    },
    {
        'classifier': [SVC()],
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf']
    },
    {
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': [3, 5, 7]
    }
]

# Realizar la búsqueda de hiperparámetros utilizando GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_resampled, y_train_resampled)

# Obtener los resultados de la búsqueda de hiperparámetros
results = grid_search.cv_results_
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Imprimir los resultados
print("Resultados de la búsqueda de hiperparámetros:")
print("Mejor modelo:", best_model)
print("Mejores parámetros:", best_params)
print("Mejor puntuación:", best_score)

# Evaluar el mejor modelo en el conjunto de prueba
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

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline

# Crear el pipeline con escalado de características, técnica de resampling y modelo de clasificación
pipeline_smote_tomek = Pipeline([
    ('scaler', StandardScaler()),
    ('resampler', SMOTE(sampling_strategy='auto')),
    ('undersampler', TomekLinks(sampling_strategy='majority')),
    ('classifier', RandomForestClassifier())
])

# Definir los parámetros del grid search
param_grid_smote_tomek = {
    'classifier__n_estimators': [50, 100, 200]
}

# Realizar la búsqueda de hiperparámetros utilizando GridSearchCV
grid_search_smote_tomek = GridSearchCV(pipeline_smote_tomek, param_grid_smote_tomek, cv=5, scoring='accuracy')
grid_search_smote_tomek.fit(X_train, y_train)

# Obtener los resultados de la búsqueda de hiperparámetros
best_model_smote_tomek = grid_search_smote_tomek.best_estimator_
best_params_smote_tomek = grid_search_smote_tomek.best_params_
best_score_smote_tomek = grid_search_smote_tomek.best_score_

# Imprimir los resultados
print("Resultados de la búsqueda de hiperparámetros para SMOTE + TomekLinks:")
print("Mejor modelo:", best_model_smote_tomek)
print("Mejores parámetros:", best_params_smote_tomek)
print("Mejor puntuación:", best_score_smote_tomek)
