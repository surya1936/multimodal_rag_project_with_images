import streamlit as st
from llm import ask_multimodal_llm
import tempfile

st.title("Multimodal RAG System")
text = st.text_input("Enter your question")
image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if st.button("Ask"):
    if text and image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(image.read())
            tmp_path = tmp_file.name

        answer = ask_multimodal_llm(text, tmp_path)
        st.write("### Answer:")
        st.write(answer)
    else:
        st.warning("Please provide both text and image.")