import streamlit as st
import joblib
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def realizar_prediccion(datos):
    # Cargar el modelo desde el archivo .pkl
    modelo = joblib.load(r"C:\Users\de969\OneDrive\Escritorio\proyecto, machine learnig\_marketing\models\mejor_modelo.pkl")

    # Realizar la prediccion con los datos proporcionados
    datos = pd.DataFrame(datos, index=[0])
    prediccion = modelo.predict(datos)

    # Devolver el resultado de la prediccion
    return prediccion[0]

def main():
    # Título de la aplicación
    st.title("Aplicación de predicción")

    # Obtener los datos de entrada del usuario
    st.subheader("Ingrese los datos:")
    datos = {}
    datos["Year_Birth"] = st.number_input("Año de nacimiento", min_value=1900, max_value=2023, value=2000)
    datos["Income"] = st.number_input("Ingreso", min_value=0, max_value=100000, value=50000)
    datos["Kidhome"] = st.number_input("Número de hijos pequeños", min_value=0, max_value=10, value=0)
    datos["Teenhome"] = st.number_input("Número de adolescentes en el hogar", min_value=0, max_value=10, value=0)
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

    # Botón para realizar la predicción
    if st.button("Realizar predicción"):
        resultado = realizar_prediccion(datos)
        st.subheader("Resultado:")
        st.write(resultado)
        


if __name__ == "__main__":
    main()







