import streamlit as st
import json
from src import NaiveBayesClassifier, BertClassifier, HFDatasetGenerator, OpenAIDatasetGenerator
import pandas as pd     
from streamlit_tags import st_tags
from collections import defaultdict
import json

from config import MODELS_FOLDER


### SETUP ###
#st.set_page_config(layout="wide")
if "data_generator_mode" not in st.session_state:
    st.session_state["data_generator_mode"] = "Generar"
if "data_generator_model_type" not in st.session_state:
    st.session_state["data_generator_model_type"] = "OpenAI"
if "context" not in st.session_state:
    st.session_state["context"] = ""
if "text_classes" not in st.session_state:
    st.session_state["text_classes"] = ["triste","alegre", "enojado"]
if "samples_per_class" not in st.session_state:
    st.session_state["samples_per_class"] = 20
if "number_of_words" not in st.session_state:
    st.session_state["number_of_words"] = 50
if "dataset_dict" not in st.session_state:
    st.session_state["dataset_dict"] = defaultdict(list)
if "model" not in st.session_state:
    st.session_state["model"] = None

### RENDER START ###
st.image("misc/banner.png")

# Select generation mode
_, center, _ = st.columns(3)
with center:
    st.session_state.data_generator_mode = st.pills(label="Elija una opcion", selection_mode="single", options=["Generar", "Subir Datos"], default=st.session_state.data_generator_mode)
st.divider()

if st.session_state.data_generator_mode == "Generar":
    # Inicializar la lista en la sesión si no existe
    st.write("## Generador de Dataset")
    st.session_state.data_generator_model_type = st.pills(label="Elija Modelo", selection_mode="single", options=["OpenAI", "Hugging Face"], default=st.session_state.data_generator_model_type)

    
    st.session_state.text_classes = st_tags( label= "#### Ingrese las clases (presione enter para agregar)", maxtags=10)
    st.write("#### Ingrese un contexto para el generador.")
    st.session_state.context = st.text_area(" ",height=200)
    st.session_state.samples_per_class = st.slider("Samples per Class", min_value=10, max_value=100, step=10, value=st.session_state.samples_per_class)
    st.session_state.number_of_words = st.slider("Number of Words", max_value=100, min_value=10, step=5, value=st.session_state.number_of_words)
    if st.button("Generar Dataset"):
        generator = OpenAIDatasetGenerator("")
        with st.spinner("Generando Dataset..."):
            st.session_state.dataset_dict = generator.generate(context= st.session_state.context,
                                        classes=st.session_state.text_classes,
                                        samples_per_class = int(st.session_state.samples_per_class / 10),
                                        number_of_words=st.session_state.number_of_words)
        st.success("Dataset Generado!!")   
elif st.session_state.data_generator_mode == "Subir Datos":
    st.write("## Subir Dataset")
    st.write("#### Suba un archivo .json con el dataset.")
    uploaded_file = st.file_uploader(label="Subir archivo", type=["json"])
    if uploaded_file is not None:
        try:
            # Read the file as a string
            file_content = uploaded_file.read().decode("utf-8")
            
            # Convert to dictionary
            st.session_state.dataset_dict = json.loads(file_content)
            st.session_state.text_classes = list(st.session_state.dataset_dict.keys())
            st.success("Archivo Subido!")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
st.divider()

if st.session_state.dataset_dict.keys():
    st.write(" ### Navigate the data:")
    st.write(f"Classes: {st.session_state.text_classes}")
    st.json(st.session_state.dataset_dict, expanded=False)
    st.divider()
    if st.button("Generar modelo"):
        if st.session_state.dataset_dict.keys():
            
            with st.spinner("Training Model..."):
                classifier = NaiveBayesClassifier(labels=st.session_state.text_classes)
                classifier.train(data=st.session_state.dataset_dict, dataset=None)
                classifier.export(dir=MODELS_FOLDER)
                st.session_state.model = classifier
            st.success("Model Trained.")
        else:
            st.warning("La lista está vacía. Agrega elementos antes de generar el modelo.")
st.divider()

if st.session_state.model is not None:
    texto_input = st.text_input("Introduce un texto para el modelo y presiona enter:")
    result = ""
    if texto_input:
        result = st.session_state.model.predict(texto_input)
    if result:
        st.write(f"## Result: {result}")

