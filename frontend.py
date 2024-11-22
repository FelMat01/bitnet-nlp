import streamlit as st
import pdb
import pandas as pd     
from streamlit_tags import st_tags
#st.set_page_config(layout="wide")
if "data_generator_mode" not in st.session_state:
        st.session_state["data_generator_mode"] = "Generar"
if "text_list" not in st.session_state:
    st.session_state["text_list"] = []
if "samples_per_class" not in st.session_state:
    st.session_state["samples_per_class"] = 20
if "number_of_words" not in st.session_state:
    st.session_state["number_of_words"] = 50
st.image("misc/banner.png")
_, col2, _ = st.columns(3)
with col2:
    st.session_state.data_generator_mode = st.pills(label = "", selection_mode="single", options=["Generar", "Subir Datos"], default="Generar")

st.divider()
if st.session_state.data_generator_mode == "Generar":
    # Inicializar la lista en la sesión si no existe
    st.write("## Generador de Dataset")
    st.session_state["text_list"] = st_tags(suggestions=["triste","alegre", "enojado"], label= "#### Ingrese las clases (presione enter para agregar)", maxtags=10)  
    st.write("#### Ingrese un contexto para el generador.")
    user_input = st.text_area("",height=200)
    st.session_state.samples_per_class = st.slider("Samples per Class", min_value=10, max_value=100, step = 10, value = st.session_state.samples_per_class)
    st.session_state.number_of_words = st.slider("Number of Words", max_value = 100, min_value = 10, step=5, value=st.session_state.number_of_words)

elif st.session_state.data_generator_mode == "Subir Datos":
    st.write("## Subir Dataset")
    st.write("#### Suba un archivo .json con el dataset.")
    st.file_uploader(label="")
    

st.divider()
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
