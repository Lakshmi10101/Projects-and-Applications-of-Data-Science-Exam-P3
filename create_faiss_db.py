from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import yaml
from langchain_openai import OpenAIEmbeddings

DB_FAISS_PATH = 'faiss_vectorstore/'


OPENAI_CONFIG_FILE = 'api_key.yaml'

with open(OPENAI_CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

apikey = config['openai']['access_key']




def load_documents(path):
    loader = DirectoryLoader(path, glob="**/*.txt", show_progress=True)
    documents = loader.load()
    
    return documents


# Create vector database
def create_vector_db():
    # Load retrieved content
    
    loader = TextLoader("museum_content.txt")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                                   chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    #embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
     #                                  model_kwargs={'device': 'cpu'})
    
    embeddings = OpenAIEmbeddings(openai_api_key = apikey)

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()
