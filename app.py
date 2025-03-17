from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Step 1: Load the PDF Manual & Extract Text 
# Load the PDF manual
pdf_path = "manual.pdf"  # Change this to your actual PDF file
loader = PyPDFLoader(pdf_path)

# Extract pages as text
documents = loader.load()
print(f"Loaded {len(documents)} pages from PDF")

# Step 2: Split the Text into Chunks
# Split text into smaller chunks (to fit within embedding limits)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks")

# Step 3: Create Embeddings & Store in ChromaDB

# Load a local embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store chunks in ChromaDB with embeddings
vector_db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db")

print("Stored chunks in ChromaDB")

# Step 4: Retrieve Relevant Chunks for a Query
# Load the vector store (in case of restarting script)
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

query = "How do I change toolbar settings?"
results = vector_db.similarity_search(query, k=3)  # Get top 3 matching chunks

for i, doc in enumerate(results):
    print(f"\nResult {i+1}: {doc.page_content}")

# Step 5: Use Ollama to Answer Questions
from langchain.chat_models import ChatOllama

# Load the Ollama LLM
llm = ChatOllama(model="mistral")  # Change to another model if needed

# Format retrieved text as context
retrieved_text = "\n\n".join([doc.page_content for doc in results])

# Prompt with RAG context
prompt = f"Use the following manual excerpts to answer:\n\n{retrieved_text}\n\nQ: {query}\nA:"

# Generate response
response = llm.predict(prompt)
print("\n💡 Ollama's Answer:\n", response)