from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone 
from dotenv import load_dotenv
import os

load_dotenv()

#PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
#PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

PINECONE_API_KEY = "pcsk_3nMiAp_LGd5TtGfh5aiWcxNNuwq2vWGy5TigJLL1CvzD6MtNpskpfwSEudwC6mnJFMzzS1"
PINECONE_API_ENV = "us-central1-gcp"

extracted_data = load_pdf("/Users/donghunshin/Documents/End-to-end-Medical-Chatbot-using-Llama2/data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name="medbud"

#Creating Embeddings for Each of The Text Chunks & storing
docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)