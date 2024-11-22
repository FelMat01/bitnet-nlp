import streamlit as st
import pdb
import pandas as pd     
from streamlit_tags import st_tags

user_input = st.text_area("Ingresa su caso de uso:", height=200)

# Mostrar el texto ingresado
st.write("Texto ingresado:", user_input)


# Inicializar la lista en la sesión si no existe
if "text_list" not in st.session_state:
    st.session_state["text_list"] = []

st.session_state["text_list"] = st_tags(suggestions=["triste","alegre", "enojado"], label= "### Ingrese las clases (presione enter para agregar)", maxtags=10)



modelo_resultado = "Modelo generado con los elementos: " + ", ".join(st.session_state["text_list"])

# Botón para generar el modelo usando la lista actual
if st.button("Generar modelo"):
    if st.session_state["text_list"]:
        resultado_modelo = procesar_modelo(st.session_state["text_list"])
        st.success(resultado_modelo)
    else:
        st.warning("La lista está vacía. Agrega elementos antes de generar el modelo.")



# Estado inicial para controlar las etapas
if "datos_generados" not in st.session_state:
    st.session_state["datos_generados"] = False
if "modelo_entrenado" not in st.session_state:
    st.session_state["modelo_entrenado"] = False

# Botón para generar datos
st.title("Pipeline de Modelo")

if not st.session_state["datos_generados"]:
    if st.button("Generar Datos"):
        st.session_state["datos_generados"] = True
        st.success("Datos generados correctamente.")
else:
    st.info("Datos ya generados.")

# Botón para entrenar el modelo (habilitado solo si los datos han sido generados)
if st.session_state["datos_generados"] and not st.session_state["modelo_entrenado"]:
    if st.button("Entrenar Modelo"):
        st.session_state["modelo_entrenado"] = True
        st.success("Modelo entrenado correctamente.")
elif st.session_state["modelo_entrenado"]:
    st.info("Modelo ya entrenado.")

# Mostrar pestaña para usar el modelo entrenado solo si está entrenado
if st.session_state["modelo_entrenado"]:
    st.write("---")  # Línea divisoria
    st.header("Uso del Modelo Entrenado")
    tab1, tab2 = st.tabs(["Input de Modelo", "Acerca del Modelo"])

    with tab1:
        # Input de texto para el modelo
        texto_input = st.text_input("Introduce un texto para el modelo:")
        if st.button("Generar Output"):
            if texto_input:
                # Modelo ficticio que genera un output dummy
                output_modelo = f"Este es un output ficticio basado en el input: '{texto_input}'"
                st.success(output_modelo)
            else:
                st.warning("Por favor, introduce un texto antes de generar el output.")

    with tab2:
        st.write("Esta pestaña proporciona información adicional sobre el modelo.")
        st.write("Por ejemplo, detalles de configuración, parámetros, etc.")
