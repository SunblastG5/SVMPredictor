import streamlit as st
from SVM import SVM

st.sidebar.title("Examen Final - Mineria de datos")
with st.sidebar.expander("Información", True):
    st.markdown("""
        # Sistema Web de prediccion de cancer de mama.
        ## Curso  
        **Mineria de Datos**
        ## Profesor: 
        **Calderón Vilca, Hugo David**
        ## Escuela Profesional:  
        **Ingeniería de Software** 
        ## Grupo 1
        * **Aguilar Salazar, Edwin Ccari**
        * **Alberto Miranda, Anderson Leandro**
        * **Arias Quispe, Alexis Enrique** 
        * **Bojorquez Suarez, Rafael Alejandro**
        * **Ramos Paredes, Roger Anthony**
    """)

st.title("Bienvenido al programa de predicción de cancer de mama")
st.write("Aqui podrá usted saber si un paciente podría tener cancer de mama a partir de la textura y el área del tumor utilizando el algoritmo de Support Vector Machine")

svm = SVM()

with st.spinner(text='En progreso'):
    svm.EntrenarModelo()
    st.success('Modelo Cargado')

texture_mean = st.number_input(label="Escriba aqui la textura del tumor", min_value=0.0)
area_mean = st.number_input(label="Escriba aqui el area del tumor", min_value=0.0)

if st.button("Predecir"):
    prediccion = svm.Prediccion(texture_mean, area_mean)
    if prediccion == 0:
        st.error("El tumor es maligno")
    elif prediccion == 1:
        st.success("El tumor es benigno")
    else:
        st.write("Predicción")




