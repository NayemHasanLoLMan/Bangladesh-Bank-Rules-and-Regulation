import os
from pathlib import Path
import time
import hashlib
from pinecone import Pinecone, ServerlessSpec
import PyPDF2
import easyocr
from pdf2image import convert_from_path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION") or os.getenv("PINECONE_ENVIRONMENT") or "us-east-1"

# Validate environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment")

print(f"Using Pinecone region: {PINECONE_REGION}")

# Configure OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072  # text-embedding-3-small dimension
BATCH_SIZE = 100


class PDFUploader:
    def __init__(self, index_name="test-singlefile-docs"):
        """
        Args:
            index_name: Name of the Pinecone index
        """
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = index_name
        self.index = None
        
        # Initialize EasyOCR with English and Bengali
        print("Initializing OCR (this may take a moment)...")
        self.ocr_reader = easyocr.Reader(['en', 'bn'], gpu=True)
        print("OCR ready!\n")

    def create_index(self):
        existing = self.pc.list_indexes().names()
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
            )
            self._wait_until_ready(timeout=20)
        self.index = self.pc.Index(self.index_name)

    def _wait_until_ready(self, timeout=20, poll=2.0):
        start = time.time()
        while True:
            desc = self.pc.describe_index(self.index_name)
            status = getattr(desc, "status", None)
            ready = getattr(status, "ready", None)
            state = getattr(status, "state", None)
            if ready is True or state == "Ready":
                return
            if time.time() - start > timeout:
                raise TimeoutError(f"Index {self.index_name} not ready after {timeout}s")
            time.sleep(poll)

    def extract_text_with_ocr(self, pdf_path):
        """Extract text using OCR for image-heavy PDFs"""
        try:
            images = convert_from_path(pdf_path, dpi=200)
            all_text = []
            
            for page_num, image in enumerate(images):
                results = self.ocr_reader.readtext(image)
                page_text = "\n".join([text for (_, text, _) in results])
                
                if page_text.strip():
                    all_text.append(page_text)
                    
            return "\n\n".join(all_text)
        except Exception as e:
            print(f" OCR extraction failed: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path):
        """Extract full text from entire PDF as single chunk"""
        all_text = []
        
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    all_text.append(page_text)
        
        combined_text = "\n\n".join(all_text)
        
        # If very little text extracted, try OCR
        if len(combined_text.strip()) < 100:
            print(f" Low text extraction, attempting OCR...")
            ocr_text = self.extract_text_with_ocr(pdf_path)
            if ocr_text:
                combined_text = ocr_text
        
        return combined_text

    def get_embedding(self, text):
        """Get embedding using OpenAI"""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    input=text,
                    model=EMBEDDING_MODEL
                )
                
                return response.data[0].embedding
                
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f" Rate limit hit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f" Embedding error after {max_retries} attempts: {e}")
                        return None
                else:
                    print(f" Embedding error: {e}")
                    return None
        
        return None

    def chunk_text(self, text, max_tokens=2048, overlap=200):
        """Split text into overlapping chunks based on token estimate"""
        # Rough estimate: 1 token ≈ 4 characters for English
        max_chars = max_tokens * 4
        overlap_chars = overlap * 4
        
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            # Try to break at sentence or word boundary
            if end < len(text):
                # Look for sentence end
                last_period = text.rfind('.', start, end)
                last_question = text.rfind('?', start, end)
                last_exclaim = text.rfind('!', start, end)
                
                break_point = max(last_period, last_question, last_exclaim)
                
                # If no sentence boundary, try word boundary
                if break_point <= start:
                    last_space = text.rfind(' ', start, end)
                    if last_space > start:
                        break_point = last_space
                    else:
                        break_point = end
                else:
                    break_point += 1  # Include the punctuation
                
                end = break_point
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap_chars
            if start < 0:
                start = end
        
        return chunks

    def generate_pdf_id(self, file_path, chunk_num=0):
        """Generate unique ID for PDF chunk"""
        file_hash = hashlib.md5(file_path.encode("utf-8")).hexdigest()[:8]
        filename = os.path.splitext(os.path.basename(file_path))[0]
        if chunk_num > 0:
            return f"{file_hash}_{filename}_chunk_{chunk_num:02d}"
        return f"{file_hash}_{filename}"

    def upload_pdf(self, pdf_path):
        """Upload PDF with chunking and overlap"""
        print(f"Processing: {os.path.basename(pdf_path)}")
        
        full_text = self.extract_text_from_pdf(pdf_path)
        if not full_text.strip():
            print(f" No text extracted\n")
            return
        
        print(f" Extracted text ({len(full_text)} characters)")
        
        # Split into overlapping chunks
        chunks = self.chunk_text(full_text)
        print(f" Created {len(chunks)} chunk{'s' if len(chunks) > 1 else ''}")
        
        vectors = []
        
        for chunk_idx, chunk in enumerate(chunks):
            emb = self.get_embedding(chunk)
            if emb is None:
                continue
            
            # Generate unique ID for each chunk
            if len(chunks) > 1:
                vid = self.generate_pdf_id(pdf_path, chunk_idx + 1)
            else:
                vid = self.generate_pdf_id(pdf_path)
            
            # Metadata with full chunk text
            metadata = {
                "source": os.path.basename(pdf_path),
                "text": chunk
            }
            
            if len(chunks) > 1:
                metadata["chunk"] = chunk_idx + 1
                metadata["total_chunks"] = len(chunks)
            
            vectors.append({
                "id": vid,
                "values": emb,
                "metadata": metadata
            })
            
            # Batch upload
            if len(vectors) >= BATCH_SIZE:
                self.index.upsert(vectors=vectors)
                print(f"  → Uploaded {len(vectors)} vectors")
                vectors = []
        
        # Upload remaining vectors
        if vectors:
            self.index.upsert(vectors=vectors)
            print(f"  → Uploaded {len(vectors)} vectors")
        
        print(f" Completed\n")

    def upload_directory(self, directory_path):
        """Upload all PDFs in directory"""
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files\n")
        
        for pdf_file in pdf_files:
            try:
                self.upload_pdf(str(pdf_file))
            except Exception as e:
                print(f" Error: {e}\n")


if __name__ == "__main__":
    uploader = PDFUploader(index_name="bangladesh-bank-docs")
    uploader.create_index()
    uploader.upload_directory(r"C:\Users\hasan\Downloads\Bangladesh Bank\documents")