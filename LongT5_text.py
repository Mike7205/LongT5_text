import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2

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
st.title("Porównanie PDF za pomocą LongT5")
uploaded_files = st.file_uploader("Wybierz dwa pliki PDF", accept_multiple_files=True, type="pdf")

if uploaded_files and len(uploaded_files) == 2:
    file1, file2 = uploaded_files
    text1 = read_pdf(file1)
    text2 = read_pdf(file2)
    
    inputs1 = tokenizer(text1, return_tensors="pt", max_length=4096, truncation=True)
    inputs2 = tokenizer(text2, return_tensors="pt", max_length=4096, truncation=True)
    
    outputs1 = model.generate(inputs1.input_ids, max_length=512)
    outputs2 = model.generate(inputs2.input_ids, max_length=512)
    
    result1 = tokenizer.decode(outputs1[0], skip_special_tokens=True)
    result2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
    
    st.subheader(f"Wynik analizy dla {file1.name}")
    st.write(result1)
    
    st.subheader(f"Wynik analizy dla {file2.name}")
    st.write(result2)
    
    with open("analysis_results.txt", "w", encoding="utf-8") as result_file:
        result_file.write(f"Analysis of {file1.name}:\n{result1}\n\n")
        result_file.write(f"Analysis of {file2.name}:\n{result2}\n\n")
    
    st.success("Wyniki analizy zostały zapisane do pliku 'analysis_results.txt'.")
    
    with open("analysis_results.txt", "r", encoding="utf-8") as file:
        st.download_button(
            label="Pobierz wyniki analizy",
            data=file,
            file_name="analysis_results.txt",
            mime="text/plain"
        )
else:
    st.warning("Proszę wgrać dokładnie dwa pliki PDF.")
