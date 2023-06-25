import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
# from dataprep.eda import create_report
from datetime import datetime
import csv
import os




marketing = pd.read_csv('data/raw/marketing_campaign.csv', sep='\t')
marketing

# Función para calcular la edad actual
def calculate_age(year):
           current_year = datetime.now().year
           age = current_year - year
           return age


# Reemplazar la columna 'Year_Birth' por las edades correspondientes
marketing['Year_Birth'] = marketing['Year_Birth'].apply(calculate_age)

# Imprimir el DataFrame actualizado
print(marketing)

m = pd.DataFrame({'Marital_Status': ['Single', 'Together', 'Married', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO']})

# Mapear el estado civil a valores binarios (1 para casado, 0 para soltero)
marketing['Marital_Status_Binary'] = marketing['Marital_Status'].map({'Single': 0, 'Together': 1, 'Married': 1, 'Divorced': 0, 'Widow': 0, 'Alone': 0, 'Absurd': 0, 'YOLO': 0})

# hijos en total de las familias
marketing['Kids'] = marketing['Kidhome'] + marketing['Teenhome']
marketing.drop(['Kidhome', 'Teenhome'], axis=1)

#funcion para cambiar la fecha a meses solo
def fixed_df(arg):
    d = {1: 'Yan', 2: 'Feb', 3: 'March', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 
         'July', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'November', 12: 'December'}

    arg = int(arg.split('-')[1])
    return d[arg]

marketing['date'] = marketing['Dt_Customer'].apply(fixed_df)

# compras en total
marketing['monto_gastado'] = marketing['MntFishProducts'] + marketing['MntFruits'] + marketing['MntGoldProds'] + marketing['MntMeatProducts'] + marketing['MntSweetProducts'] + marketing['MntWines']

# filas incesarias
marketing = marketing.drop(['Kidhome', 'Teenhome', 'Z_CostContact'], axis=1)
# quitar nulos de income 
median_income = marketing['Income'].median()
marketing['Income'].fillna(median_income, inplace=True)
print(marketing)

marketing.to_csv('app\marketing_limpio.csv',index= False)