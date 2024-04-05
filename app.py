import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def process_pdf_and_chat(pdf_path, question):
    # Load PDF document
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splitted_chunks = text_splitter.split_documents(docs)

    # Extract embeddings from chunks
    # embeddings_model = HuggingFaceEmbeddings("microsoft/layoutlm-base-uncased")
    embeddings_model = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(splitted_chunks, embeddings_model)

    # Initialize chat model
    chat_model = ChatOpenAI()

    # Initialize retrieval question answering system
    retrieval = vectordb.as_retriever()
    retrival_chain = RetrievalQA.from_chain_type(chat_model, retriever=retrieval)

    # Ask the question and get the answer
    answer = retrival_chain.run(question)

    return answer

# Define Gradio interface
inputs = [
    gr.File(label="Upload PDF"),
    gr.Textbox(label="Question")
]
outputs = gr.Textbox(label="Answer")

gr.Interface(fn=process_pdf_and_chat, inputs=inputs, outputs=outputs, title="PDF Chatbot", description="Ask questions about a PDF document").launch(share=True,debug=True)
