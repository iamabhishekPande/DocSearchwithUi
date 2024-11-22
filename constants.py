# import os
# # from chromadb.config import Settings
# from langchain_community.document_loaders import  CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader, UnstructuredPowerPointLoader

# ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
# SOURCE_DIRECTORY = os.path.join(ROOT_DIRECTORY, "source_documents")
# PERSIST_DIRECTORY = os.path.join(ROOT_DIRECTORY, "DB")

# # Can be changed to a specific number
# INGEST_THREADS = os.cpu_count() or 8

# # # Define the updated Chroma settings
# # CHROMA_SETTINGS = {
# #     "database": {
# #         "implementation": "duckdb+parquet",
# #         "persist_directory": PERSIST_DIRECTORY,
# #         "anonymized_telemetry": False
# #     }
# # }

# DOCUMENT_MAP = {
#     ".txt": TextLoader,
#     ".md": TextLoader,
#     ".py": TextLoader,
#     ".pdf": PDFMinerLoader,
#     ".csv": CSVLoader,
#     ".xls": UnstructuredExcelLoader,
#     ".xlsx": UnstructuredExcelLoader,
#     ".docx": Docx2txtLoader,
#     ".doc": Docx2txtLoader,
#     ".ppt": UnstructuredPowerPointLoader,
#     ".pptx": UnstructuredPowerPointLoader
# }

import os
from langchain_community.document_loaders import  CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader, UnstructuredPowerPointLoader


# Root directory
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Source directory for uploaded documents
SOURCE_DIRECTORY = os.path.join(ROOT_DIRECTORY, "source_documents")

# Persistent storage directory for Chroma or other databases
PERSIST_DIRECTORY = os.path.join(ROOT_DIRECTORY, "DB")

# Number of threads for document ingestion (defaults to CPU count or 8)
INGEST_THREADS = os.cpu_count() or 8

# Mapping of file extensions to their respective document loaders
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".pptx": UnstructuredPowerPointLoader
}
# Optional: Settings for Chroma (if you choose to uncomment and use them)
# from chromadb.config import Settings
# CHROMA_SETTINGS = Settings(
#     chroma_db_impl="duckdb+parquet",
#     persist_directory=PERSIST_DIRECTORY,
#     anonymized_telemetry=False
# )

# Ensure necessary directories exist
os.makedirs(SOURCE_DIRECTORY, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
