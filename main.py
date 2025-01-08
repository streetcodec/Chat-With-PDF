import streamlit as st
from PyPDF2 import PdfReader
import fitz
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # used for general purpose hosting


genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
            st.write(text)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    # The prompt which goes to the LLM
    prompt_template = """
    Answer the question in excruciating detail using the provided context, provide all the possible details, point wise. and if the answer is not in
    provided context say, "Well the question is out the syllabus man!", don't provide the wrong or out of context answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4) # temperature, model and pother attributes can be fiddled with too fine tune the use for specific use case. Maybe the option can be given to user to manipulate as they see fit.


    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def highlight_text_in_pdf(pdf_docs, text_to_highlight, output_name="highlighted.pdf"):

    text_to_highlight = text_to_highlight.split() # spliting to each individual strings.
    stop_words = set(stopwords.words('english')) # collects redundant words.
     
    filtered_words = [word for word in text_to_highlight if word.lower() not in stop_words and len(word) > 1] # removes stopwords, and single letters.

    pdf_document = fitz.open(stream= pdf_docs.read(), filetype="pdf")
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        for text in filtered_words: 
            instances = page.search_for(text)
            #st.write("Highlighting : ", text) # To find what is being highlighted
            for inst in instances:
                page.add_highlight_annot(inst)  # Highlights
            
    
    output_stream = io.BytesIO()
    pdf_document.save(output_stream)
    pdf_document.close()
    output_stream.seek(0)
    return output_stream

def user_input(user_question, pdf_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    ResText = response["output_text"]
    st.write("Here you go, ")
    st.write(ResText) # Final answer to the query 

    # Highlight the relevant text in the PDF
    if pdf_docs:
        for pdf in pdf_docs:
            highlighted_pdf = highlight_text_in_pdf(pdf, ResText)
            st.download_button(
                label="Download Highlighted PDF",
                data=highlighted_pdf,
                file_name="highlighted.pdf",
                mime="application/pdf",
            )

def main():
    st.set_page_config("String Venture OA")
    st.header("Chat with PDF (Open the SideBar to upload)")

    user_question = st.text_input("Ask me anything now! But it better be from the syllabus.")
    top_bar = st.columns([3, 6, 3])
    with top_bar[1]:
        pdf_docs = st.file_uploader("Upload your PDF Files and submit it", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Learning faster than you ever could :) "):
                st.write("This is the extracted text.")
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Ask me anything!")

    if user_question:
        user_input(user_question, pdf_docs)

if __name__ == "__main__":
    main()
