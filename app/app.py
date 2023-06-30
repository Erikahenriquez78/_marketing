
import pickle
from pathlib import Path
import streamlit as st
import joblib
import pandas as pd
import warnings
import os

def realizar_prediccion(datos, modelo):
    # Realizar la predicción con los datos proporcionados
    datos = pd.DataFrame(datos, index=[0])
    prediccion = modelo.predict(datos)
       # Devolver la frase personalizada según la predicción
    if prediccion[0] == 0:
        return "No participa en la campaña"
    elif prediccion[0] == 1:
        return "Sí participa en la campaña"
    else:
        return "No se puede determinar la participación"

    
    return prediccion[0]

def main():
    # Título de la aplicación
    st.image("https://media.vozpopuli.com/2022/07/alimentos-saludables-incluir-dieta-diaria.jpg",width=800)
    st.title("Aplicación de predicción")
    
  

    # Obtener los datos de entrada del usuario
   # Agregar imagen relacionada con alimentos saludables
   

    
    st.subheader("Ingrese los datos:")
    datos = {}
    datos["Year_Birth"] = st.slider("Edad", min_value=0, max_value=130, value=50)
    datos["Income"] = st.slider("Ingreso", min_value=0, max_value=100000, value=1000)
    datos["Recency"] = st.slider("Días desde la última compra", min_value=0, max_value=365, value=5)
    datos["MntWines"] = st.slider("Gasto en vino", min_value=0, max_value=10000, value=500)
    datos["MntFruits"] = st.slider("Gasto en frutas", min_value=0, max_value=10000, value=200)
    datos["MntMeatProducts"] = st.slider("Gasto en carne", min_value=0, max_value=10000, value=300)
    datos["MntFishProducts"] = st.slider("Gasto en pescado", min_value=0, max_value=10000, value=120)
    datos["MntSweetProducts"] = st.slider("Gasto en dulces", min_value=0, max_value=10000, value=120)
    datos["MntGoldProds"] = st.slider("Gasto en productos de oro", min_value=0, max_value=10000, value=200)
    datos["NumDealsPurchases"] = st.slider("Número de compras con descuento", min_value=0, max_value=100, value=1)
    datos["NumWebPurchases"] = st.slider("Número de compras por web", min_value=0, max_value=100, value=3)
    datos["NumCatalogPurchases"] = st.slider("Número de compras por catálogo", min_value=0, max_value=100, value=3)
    datos["NumStorePurchases"] = st.slider("Número de compras en tienda física", min_value=0, max_value=100, value=2)
    datos["NumWebVisitsMonth"] = st.slider("Número de visitas web al mes", min_value=0, max_value=100, value=3)
    
    

    
    frutas = st.checkbox("Frutas")
    verduras = st.checkbox("Verduras")
    proteinas = st.checkbox("Proteínas")
    
        
    modelo_path = r"C:\Users\de969\OneDrive\Escritorio\_marketing\models\modelo1.pkl"
    modelo = joblib.load(modelo_path)

    # Botón para realizar la predicción
    if st.button("Realizar predicción"):
        resultado = realizar_prediccion(datos, modelo)
        st.subheader("Resultado:")
        st.write(resultado)
        
        

if __name__ == "__main__":
    main()








