import streamlit as st
from docx import Document
import pdfplumber
from PIL import Image
import pytesseract
from transformers import pipeline

# AI modellar
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Fayldan matn ajratish
def extract_text(file):
    if file.name.endswith('.txt'):
        return file.read().decode('utf-8')
    elif file.name.endswith('.docx'):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file.name.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
    elif file.name.endswith(('.png', '.jpg', '.jpeg')):
        return pytesseract.image_to_string(Image.open(file))
    else:
        return "Noto‘g‘ri format"

# Streamlit interfeysi
st.title("🧠 AI DocReader (OCR + Q&A)")
uploaded_file = st.file_uploader("Fayl yuklang (.txt, .pdf, .docx, .jpg)")

if uploaded_file:
    text = extract_text(uploaded_file)
    st.subheader("📄 Hujjat matni:")
    st.text_area("Matn", text, height=200)

    if st.button("📝 Mazmun chiqarish"):
        summary = summarizer(text[:1000])[0]['summary_text']
        st.success(f"Mazmun: {summary}")

    question = st.text_input("❓ Savolingizni yozing:")
    if question:
        answer = qa_model(question=question, context=text[:1000])['answer']
        st.info(f"Javob: {answer}")
