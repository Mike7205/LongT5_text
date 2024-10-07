import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
import os

# Wybierz model LongT5
model_name = "google/long-t5-tglobal-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Funkcja do odczytu tekstu z PDF
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

# Interfejs użytkownika Streamlit
st.title("Analiza PDF za pomocą LongT5")
uploaded_files = st.file_uploader("Wybierz pliki PDF", accept_multiple_files=True, type="pdf")

if uploaded_files:
    st.write("Bieżący katalog roboczy:", os.getcwd())  # Dodano debugowanie
    with open("analysis_results.txt", "w", encoding="utf-8") as result_file:
        for uploaded_file in uploaded_files:
            text = read_pdf(uploaded_file)
            inputs = tokenizer(text, return_tensors="pt", max_length=4096, truncation=True)
            outputs = model.generate(inputs.input_ids, max_length=512)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            st.subheader(f"Wynik analizy dla {uploaded_file.name}")
            st.write(result)
            
            result_file.write(f"Analysis of {uploaded_file.name}:\n{result}\n\n")

    st.success("Wyniki analizy zostały zapisane do pliku 'analysis_results.txt'.")
    
    # Dodaj przycisk do pobierania pliku
    with open("analysis_results.txt", "r", encoding="utf-8") as file:
        st.download_button(
            label="Pobierz wyniki analizy",
            data=file,
            file_name="analysis_results.txt",
            mime="text/plain"
        )
