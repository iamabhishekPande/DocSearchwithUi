import os
import logging
import streamlit as st
from ingestion.ingest import (load_single_document, load_document_batch, load_documents, split_documents)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from constants import (
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)
from langchain_community.embeddings import HuggingFaceEmbeddings



os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logger = logging.getLogger(__name__)

# Constants
UPLOAD_FOLDER = SOURCE_DIRECTORY
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'csv', 'md', 'ppt', 'pptx'}

# Function to check allowed file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to convert document to dictionary
def document_to_dict(document):
    return {
        "page_content": document.page_content,
        "metadata": document.metadata
    }

# Streamlit UI
st.title("Dhyey DocSearch")

# File upload functionality
uploaded_file = st.file_uploader("Upload a file", type=ALLOWED_EXTENSIONS)

if uploaded_file:
    # Save the uploaded file
    file_path = os.path.join(SOURCE_DIRECTORY, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully!")

# Ask a question functionality
st.subheader("Ask a question")

question = st.text_input("Enter your question:")

if question:
    try:
        # Load documents
        documents = load_documents("source_documents")
        
        # Split documents
        text_documents, python_documents = split_documents(documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
        )
        texts = text_splitter.split_documents(text_documents)
        texts.extend(python_splitter.split_documents(python_documents))
        
        # Initialize embeddings
        #model_path = "/app/models/Embeddings/models--sentence-transformers--all-MiniLM-L6-v2"
        model_path = r"D:\StableGPT\models\Embeddings\models--sentence-transformers--all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_path, model_kwargs={'device': 'cuda'})
        
        # Create Chroma vector store
        vectordb = Chroma.from_documents(texts, embeddings,persist_directory=PERSIST_DIRECTORY)
        retriever = vectordb.as_retriever()

        # Set up prompt template
        assistant_prompt_template = """
            SYSTEM = "system"
            USER = "user"
            ASSISTANT = "assistant"
            ...
            {context}
            {history}
            Question: {question}
            Helpful Answer:
        """
        prompt = PromptTemplate(input_variables=["question", "context", "history"], template=assistant_prompt_template)
        
        # Initialize conversation memory
        memory = ConversationBufferMemory(input_key="question", memory_key="history")
        
        # Initialize language model
        llm = CTransformers(model=r"D:\StableGPT\models\Llama 2\llama-2-13b-chat.ggmlv3.q2_K.bin",#model="/app/models/Llama 2/llama-2-13b-chat.ggmlv3.q2_K.bin",
                            model_type="llama",
                            config={'context_length': 4000},
                            temperature=0.3, device='cuda')

        # Initialize QA system
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True, chain_type_kwargs={"prompt": prompt, "memory": memory})
        
        # Get answer
        response = qa(question)
        
        # Process response
        query = response.get('query', 'No query found')
        result = response.get('result', 'No result found')
        source_documents = [document_to_dict(doc) for doc in response.get('source_documents', [])]
        
        # Display result
        st.write(f"Query: {query}")
        st.write(f"Answer: {result}")
        
        if source_documents:
            st.write("Source Documents:")
            for doc in source_documents:
                st.write(doc)

    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        st.error(f"Error: {str(e)}")