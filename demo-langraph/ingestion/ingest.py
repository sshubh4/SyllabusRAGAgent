import os
import pickle
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = "data/uploads"
INDEX_PATH = "embeddings/syllabus.index"
DOCS_PATH = "embeddings/docs.pkl"

def main():
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} directory does not exist")
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Please add PDF files to {DATA_DIR}")
        return
    
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    docs = []
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}")
        return
    
    for file in pdf_files:
        filepath = os.path.join(DATA_DIR, file)
        try:
            
            reader = PdfReader(filepath)
            for page in reader.pages:
                text = page.extract_text()
                if text.strip():
                    docs.append(text)
        except Exception as e:
            print(f"Error processing")
            continue
    
    if not docs:
        print("No text extracted from pdf")
        return
    
    embeddings = model.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    if index.ntotal != len(docs):
        raise RuntimeError(
            f"Index creation failed, index has {index.ntotal} vectors "
        )
    
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved FAISS index to {INDEX_PATH} ({index.ntotal} vectors, {index.d} dimensions)")
    
    #save documents
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)
    
    
    # Verify 
    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(f"Index file was not created")
    if not os.path.exists(DOCS_PATH):
        raise RuntimeError(f"Documents file was not created")
    
if __name__ == "__main__":
    main()

