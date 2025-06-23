import time
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter

def create_vector_store():
    # 1. Load markdown files
    loader = DirectoryLoader('/docs', glob="**/*.md", show_progress=True)
    docs = loader.load()
    print(f"📂 Loaded {len(docs)} documents from /docs")

    # 2. Split with markdown awareness
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    split_docs = []
    
    for doc in docs:
        split_docs.extend(markdown_splitter.split_text(doc.page_content, metadata=doc.metadata))
    
    print(f"✂️ Split into {len(split_docs)} chunks")

    # 3. Create vector store
    embeddings = OllamaEmbeddings(model="llama3.1:8b")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    
    # 4. Save index
    vector_store.save_local("/app/faiss_index")
    print(f"💾 Saved vector store with {vector_store.index.ntotal} embeddings")
    
    # 5. Verification test
    test_query = "Docker volume configuration"
    results = vector_store.similarity_search(test_query, k=1)
    print(f"✅ Test query '{test_query}' returned: {results[0].metadata['source']}")
    
    return vector_store

if __name__ == "__main__":
    print("🚀 Starting vectorization process...")
    create_vector_store()
    print("🎉 Vectorization completed successfully!")
