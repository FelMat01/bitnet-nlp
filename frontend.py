import streamlit as st
import pdb
import pandas as pd     



def write_context_input():

	user_input = st.text_area("Ingresa su caso de uso:", height=200)

	# Mostrar el texto ingresado
	st.write("Texto ingresado:", user_input)

def add_and_del_classes():
	# Inicializar la lista en la sesión si no existe
	if "text_list" not in st.session_state:
	    st.session_state["text_list"] = []

	# Cuadro de entrada de texto y botón para agregar texto a la lista
	new_text = st.text_input("Ingresa un nuevo elemento:")
	if st.button("Agregar a la lista"):
	    if new_text:
	        st.session_state["text_list"].append(new_text)
	        st.success(f"Elemento '{new_text}' agregado a la lista")
	    else:
	        st.warning("El cuadro de texto está vacío, ingresa un texto")

	# Mostrar la lista con botones de eliminación para cada elemento
	st.write("### Lista de elementos:")
	for i, item in enumerate(st.session_state["text_list"]):
	    col1, col2 = st.columns([0.85, 0.15])  # Dividir en columnas para el botón de eliminación
	    with col1:
	        st.write(f"- {item}")
	    with col2:
	        # Eliminar el elemento cuando el botón se presiona
	        if st.button("Eliminar", key=f"delete_{i}"):
	            st.session_state["text_list"].pop(i)
	            st.rerun()  # Forzar la recarga de la app después de eliminar un elemento
	            break  # Salir del bucle para evitar errores de índice


def procesar_modelo(lista):
    # Esta es solo una función de ejemplo que combina los elementos de la lista en una sola cadena.
    # Puedes reemplazarla por el procesamiento real que deseas hacer.
    modelo_resultado = "Modelo generado con los elementos: " + ", ".join(lista)
    return modelo_resultado
def generar_modelo():
	# Botón para generar el modelo usando la lista actual
	if st.button("Generar modelo"):
	    if st.session_state["text_list"]:
	        resultado_modelo = procesar_modelo(st.session_state["text_list"])
	        st.success(resultado_modelo)
	    else:
	        st.warning("La lista está vacía. Agrega elementos antes de generar el modelo.")

write_context_input()
add_and_del_classes()
generar_modelo()


## boton generar datos
## boton entrenar modelo (no se puede entrenar modelo sin primero generar datos)
## una vez entrenado el modelo se genera una nueva pestaña que permite dar el input del modelo (texto)...
## seguido de un boton que permite generar un output con el modelo entrenado
## recordar que no quiero que hagas nada con un modelo, solo quiero que hagas la interfaz como si el modelo existiera (usa un modelo dummy que no hace nada)