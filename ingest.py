# # # ingest.py
# # import os, glob, faiss, pickle, pdfplumber, nltk
# # from tqdm import tqdm
# # from nltk.tokenize import sent_tokenize
# # from sentence_transformers import SentenceTransformer

# # nltk.download('punkt')

# # DATA_DIR = "data/docs"
# # INDEX_DIR = "data/faiss_index"
# # os.makedirs(INDEX_DIR, exist_ok=True)

# # model = SentenceTransformer("all-MiniLM-L6-v2")

# # def extract_text(path):
# #     text = ""
# #     with pdfplumber.open(path) as pdf:
# #         for page in pdf.pages:
# #             t = page.extract_text()
# #             if t: text += t + "\n"
# #     return text

# # def chunk_text(text, size=5):
# #     sents = sent_tokenize(text)
# #     return [" ".join(sents[i:i+size]) for i in range(0, len(sents), size)]

# # def build_index():
# #     docs = []
# #     for file in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
# #         text = extract_text(file)
# #         chunks = chunk_text(text)
# #         for i, chunk in enumerate(chunks):
# #             docs.append({"source": os.path.basename(file), "text": chunk})

# #     embeddings = model.encode([d["text"] for d in docs], convert_to_numpy=True)
# #     index = faiss.IndexFlatL2(embeddings.shape[1])
# #     index.add(embeddings)

# #     faiss.write_index(index, f"{INDEX_DIR}/buc_index.faiss")
# #     with open(f"{INDEX_DIR}/meta.pkl", "wb") as f:
# #         pickle.dump(docs, f)

# #     print(f"✅ Indexed {len(docs)} chunks from {len(glob.glob(DATA_DIR+'/*.pdf'))} PDFs")

# # if __name__ == "__main__":
# #     build_index()





# import os
# import pdfplumber
# import numpy as np
# import faiss
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# import nltk

# nltk.download('punkt')

# # Load .env if needed
# load_dotenv()

# PDF_FOLDER = "data/pdfs"
# INDEX_FOLDER = "faiss_index"

# os.makedirs(INDEX_FOLDER, exist_ok=True)

# model = SentenceTransformer('all-MiniLM-L6-v2')

# def extract_text_from_pdfs(folder):
#     docs = []
#     for file in os.listdir(folder):
#         if file.endswith(".pdf"):
#             with pdfplumber.open(os.path.join(folder, file)) as pdf:
#                 text = ""
#                 for page in pdf.pages:
#                     text += page.extract_text() or ""
#                 if text.strip():
#                     docs.append(text)
#     return docs

# def build_index():
#     documents = extract_text_from_pdfs(PDF_FOLDER)

#     if not documents:
#         print("❌ No text found in PDFs. Check your PDF files.")
#         return

#     print(f"✅ Loaded {len(documents)} documents")

#     embeddings = model.encode(documents, show_progress_bar=True)
#     embeddings = np.array(embeddings).astype("float32")

#     if embeddings.size == 0:
#         print("❌ No embeddings generated. Check your text extraction.")
#         return

#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(embeddings)

#     faiss.write_index(index, os.path.join(INDEX_FOLDER, "index.faiss"))
#     print("✅ FAISS index created and saved successfully!")

# if __name__ == "__main__":
#     build_index()






import os
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

PDF_FOLDER = "data/pdfs"
INDEX_FOLDER = "faiss_index"
os.makedirs(INDEX_FOLDER, exist_ok=True)

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdfs(folder):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            with pdfplumber.open(os.path.join(folder, file)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                if text.strip():
                    docs.append({"source": file, "text": text})
    return docs

def build_index():
    documents = extract_text_from_pdfs(PDF_FOLDER)
    if not documents:
        print("❌ No text found in PDFs. Check your PDF files.")
        return

    print(f"✅ Loaded {len(documents)} document(s)")

    embeddings = MODEL.encode([d["text"] for d in documents], show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, os.path.join(INDEX_FOLDER, "index.faiss"))
    with open(os.path.join(INDEX_FOLDER, "meta.pkl"), "wb") as f:
        import pickle
        pickle.dump(documents, f)

    print("✅ FAISS index created and saved successfully!")

if __name__ == "__main__":
    build_index()
