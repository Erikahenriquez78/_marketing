import streamlit as st
import joblib
import pandas as pd
import warnings

import pickle
from pathlib import Path

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

# # Leer el modelo desde el archivo Pickle

# modelo = read_pickle("../models/mejor_modelo1.pkl")

# warnings.filterwarnings("ignore")

# def realizar_prediccion(datos):
#     # Cargar el modelo desde el archivo .pkl
#     modelo = joblib.load("../models/modelo1.pkl")
#     # modelo = read_pickle("../_marketing/models/mejor_modelo1.pkl")
#     # Realizar la prediccion con los datos proporcionados
#     datos = pd.DataFrame(datos, index=[0])
#     prediccion = modelo.predict(datos)

#     # Devolver el resultado de la prediccion
#     return prediccion[0]

# def main():
#     # Título de la aplicación
#     st.title("Aplicación de predicción")

#     # Obtener los datos de entrada del usuario
#     st.subheader("Ingrese los datos:")
#     datos = {}
#     datos["Year_Birth"] = st.number_input("Año de nacimiento", min_value=1900, max_value=2023, value=2000)
#     datos["Income"] = st.number_input("Ingreso", min_value=0, max_value=100000, value=50000)
#     datos["Recency"] = st.number_input("Días desde la última compra", min_value=0, max_value=365, value=30)
#     datos["MntWines"] = st.number_input("Gasto en vino", min_value=0, max_value=10000, value=500)
#     datos["MntFruits"] = st.number_input("Gasto en frutas", min_value=0, max_value=10000, value=200)
#     datos["MntMeatProducts"] = st.number_input("Gasto en carne", min_value=0, max_value=10000, value=300)
#     datos["MntFishProducts"] = st.number_input("Gasto en pescado", min_value=0, max_value=10000, value=100)
#     datos["MntSweetProducts"] = st.number_input("Gasto en dulces", min_value=0, max_value=10000, value=100)
#     datos["MntGoldProds"] = st.number_input("Gasto en productos de oro", min_value=0, max_value=10000, value=200)
#     datos["NumDealsPurchases"] = st.number_input("Número de compras con descuento", min_value=0, max_value=100, value=1)
#     datos["NumWebPurchases"] = st.number_input("Número de compras por web", min_value=0, max_value=100, value=1)
#     datos["NumCatalogPurchases"] = st.number_input("Número de compras por catálogo", min_value=0, max_value=100, value=1)
#     datos["NumStorePurchases"] = st.number_input("Número de compras en tienda física", min_value=0, max_value=100, value=1)
#     datos["NumWebVisitsMonth"] = st.number_input("Número de visitas web al mes", min_value=0, max_value=100, value=1)

#     # Botón para realizar la predicción
#     if st.button("Realizar predicción"):
#         resultado = realizar_prediccion(datos)
#         st.subheader("Resultado:")
#         st.write(resultado)
        


# if __name__ == "__main__":
#     main()


import streamlit as st
import joblib
import pandas as pd
import warnings
import os

def realizar_prediccion(datos, modelo):
    # Realizar la predicción con los datos proporcionados
    datos = pd.DataFrame(datos, index=[0])
    prediccion = modelo.predict(datos)

    # Devolver el resultado de la prediccion
    print('Daria respuesta a nuestra campaña?')
    return prediccion[0]

def main():
    # Título de la aplicación
    st.title("Aplicación de predicción")

    # Obtener los datos de entrada del usuario
    st.subheader("Ingrese los datos:")
    datos = {}
    datos["Year_Birth"] = st.number_input("edad", min_value=0, max_value=130, value=50)
    datos["Income"] = st.number_input("Ingreso", min_value=0, max_value=100000, value=1000)
    datos["Recency"] = st.number_input("Días desde la última compra", min_value=0, max_value=365, value=30)
    datos["MntWines"] = st.number_input("Gasto en vino", min_value=0, max_value=10000, value=500)
    datos["MntFruits"] = st.number_input("Gasto en frutas", min_value=0, max_value=10000, value=200)
    datos["MntMeatProducts"] = st.number_input("Gasto en carne", min_value=0, max_value=10000, value=300)
    datos["MntFishProducts"] = st.number_input("Gasto en pescado", min_value=0, max_value=10000, value=100)
    datos["MntSweetProducts"] = st.number_input("Gasto en dulces", min_value=0, max_value=10000, value=100)
    datos["MntGoldProds"] = st.number_input("Gasto en productos de oro", min_value=0, max_value=10000, value=200)
    datos["NumDealsPurchases"] = st.number_input("Número de compras con descuento", min_value=0, max_value=100, value=1)
    datos["NumWebPurchases"] = st.number_input("Número de compras por web", min_value=0, max_value=100, value=1)
    datos["NumCatalogPurchases"] = st.number_input("Número de compras por catálogo", min_value=0, max_value=100, value=1)
    datos["NumStorePurchases"] = st.number_input("Número de compras en tienda física", min_value=0, max_value=100, value=1)
    datos["NumWebVisitsMonth"] = st.number_input("Número de visitas web al mes", min_value=0, max_value=100, value=1)

    # Cargar el modelo desde el archivo .pkl
    # modelo_path = os.path.join("models", "modelo1.pkl")
    # modelo = joblib.load(modelo_path)
        # Cargar el modelo desde el archivo .pkl
    modelo_path = r"C:\Users\de969\OneDrive\Escritorio\_marketing\models\modelo1.pkl"
    modelo = joblib.load(modelo_path)

    # Botón para realizar la predicción
    if st.button("Realizar predicción"):
        resultado = realizar_prediccion(datos, modelo)
        st.subheader("Resultado:")
        st.write(resultado)

if __name__ == "__main__":
    main()








