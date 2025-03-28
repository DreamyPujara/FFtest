import argparse
import os
import shutil
import json
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from pathlib import Path
import re
import urllib

# Define paths
CHROMA_PATH = "chroma"
DATA_PATH_ABSENCE = "data/absence_FF"  # Folder for absence-related documents
DATA_PATH_COMPENSATION = "data/compensation_FF"  # Folder for compensation-related documents
DATA_PATH_BENEFITS = "data/benefits_FF"  # Folder for benefits-related documents
DATA_PATH_ORC = "data/orc_FF"  # Folder for orc-related documents
# Mapping for local to web links
local_to_web_mapping = {
    "data\\ffug.pdf": "https://docs.oracle.com/cd/A60725_05/pdf/ffug.pdf",
    "data\\ffg2.pdf": "https://docs.oracle.com/en/cloud/saas/human-resources/24d/oapff/administering-fast-formulas.pdf",
    # Add more mappings as needed
}

def populate():
    # Create directories if they don't exist
    os.makedirs(DATA_PATH_ABSENCE, exist_ok=True)
    os.makedirs(DATA_PATH_COMPENSATION, exist_ok=True)
    os.makedirs(DATA_PATH_BENEFITS, exist_ok=True)
    os.makedirs(DATA_PATH_ORC, exist_ok=True)
    
    # Check if the database should be cleared
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Load and process documents from both folders
    documents_absence = load_documents(DATA_PATH_ABSENCE)
    documents_compensation = load_documents(DATA_PATH_COMPENSATION)
    documents_benefits = load_documents(DATA_PATH_BENEFITS)
    documents_orc = load_documents(DATA_PATH_ORC)
    
    # Combine documents from both folders
    all_documents = documents_absence + documents_compensation+documents_benefits+documents_orc
    
    if all_documents:
        chunks = split_documents(all_documents)
        i = 0
        while i < len(chunks):
            if i + 1 < len(chunks) and not chunks[i + 1].page_content.startswith(' ##'):
                chunks[i].page_content += chunks[i + 1].page_content
                chunks.pop(i + 1)
                i -= 1
            i += 1
        add_to_chroma(chunks, "combined_collection")  # Use a single collection for all documents
    
    # Load prompts from JSON files in both folders
    load_prompts(DATA_PATH_ABSENCE)
    load_prompts(DATA_PATH_COMPENSATION)
    load_prompts(DATA_PATH_BENEFITS)
    load_prompts(DATA_PATH_ORC)

def load_documents(data_path):
    """Load documents from the specified folder."""
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()

def split_documents(documents: list[Document]):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=True,
        separators=[" ##"]
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document], collection_name):
    """Add chunks to the Chroma database."""
    db = Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    """Calculate unique IDs for chunks."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if source:
            base_path = "/hackathon"
            file_path = os.path.join(base_path, source).replace("\\", "/")
            local_file_path = urllib.parse.quote(file_path)
            page_link = f"file:///D:{local_file_path}#page={page}"
            chunk.metadata["page_link"] = page_link
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def load_prompts(data_path):
    """Load prompts from JSON files in the specified folder."""
    try:
        docs_dir = Path(data_path)
        json_files = list(docs_dir.glob("*.json"))
        
        if not json_files:
            print(f"Error: No JSON files found in {data_path}")
            return
        
        for json_file in json_files:
            try:
                with open(json_file, "r") as file:
                    documents_data = json.load(file)
                    print(f"Loaded {len(documents_data)} documents from {json_file.name}")
                
                # Recreate Document objects
                promptDocuments = [
                    Document(page_content=f"**question**: {doc['Description']}\n\n**Formula_Name**: {doc['BASE_FORMULA_NAME']}, **LEGISLATIVE_DATA_GROUP**: {doc['LEGISLATIVE_DATA_GROUP']}\n\n **FastFormula**:: ```{doc['FORMULA_TEXT']}```",
                            metadata={"source": "dataset", "FORMULA_TYPE_NAME": doc['FORMULA_TYPE_NAME']})
                    for doc in documents_data
                ]
                
                # Add to Chroma database with filename as argument
                collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', json_file.stem)
                add_to_chroma(promptDocuments, collection_name)
                print(f"Successfully added documents from {json_file.name} to Chroma database")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON format in {json_file.name}")
            except Exception as e:
                print(f"Error processing {json_file.name}: {str(e)}")
    except Exception as e:
        print(f"Error loading prompts: {str(e)}")

def clear_database():
    """Clear the Chroma database and data folders."""
    db = Chroma(
        collection_name="combined_collection",
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    db.delete_collection()
    if os.path.exists(DATA_PATH_ABSENCE):
        shutil.rmtree(DATA_PATH_ABSENCE, onerror=on_error)
        print(f"Deleted: {DATA_PATH_ABSENCE}")
    if os.path.exists(DATA_PATH_COMPENSATION):
        shutil.rmtree(DATA_PATH_COMPENSATION, onerror=on_error)
        print(f"Deleted: {DATA_PATH_COMPENSATION}")
    if os.path.exists(DATA_PATH_BENEFITS):
        shutil.rmtree(DATA_PATH_BENEFITS, onerror=on_error)
        print(f"Deleted: {DATA_PATH_BENEFITS}")
    if os.path.exists(DATA_PATH_ORC):
        shutil.rmtree(DATA_PATH_ORC, onerror=on_error)
        print(f"Deleted: {DATA_PATH_ORC}")
        
        

def on_error(func, path, exc_info):
    """Handle errors during file deletion."""
    import stat
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

# Run the populate function
populate()