from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from model import best_model,X_test,y_test
best_model.predict(X_test)


y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

import pickle

# Guardar el mejor modelo en un archivo pickle
with open('mejor_modelo.pkl', 'wb') as file:
    pickle.dump(best_model, file)

