import pandas as pd
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

import os
import re
import shutil

# Replace with your actual file path
file_path = r"D:\celebrity tea\venv\tiktok_comments.csv"  # Windows example

# API key for HuggingFace
api_key = ""

# Enhanced prompt template for semantic understanding
template = """
You are an assistant that answers questions based on TikTok comments about celebrity rumors.
Use the following TikTok comments as your knowledge source:

{context}

Question: {question}

Important instructions:
1. Base your answer ONLY on the provided comments
2. Make it clear that this information comes from TikTok comments and may be rumors
3. Include how popular the comments were (likes count) when relevant
4. Do not make claims beyond what's stated in the comments
5. Ignore spelling mistakes in names of people
6. Focus on the semantic meaning of the question, not just keywords
7. If the comments don't contain relevant information to the question, state this clearly
8. Analyze patterns across multiple comments to identify consistent rumors

Answer:
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Function to deduplicate documents
def deduplicate_documents(documents):
    """Function to deduplicate documents based on content."""
    seen_content = set()
    deduplicated_docs = []
    
    for doc in documents:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            deduplicated_docs.append(doc)
    
    return deduplicated_docs

# Enhanced comment cleaning
def clean_comment(text):
    if pd.isna(text):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove emojis and special characters (basic version)
    text = re.sub(r'[^\w\s,.!?]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

# Function to setup the database with improved embeddings
def setup_database(file_path):
    # Verify file exists
    if not os.path.exists(file_path):
        print(f"File not found at: {file_path}")
        return False
    
    print(f"File found at: {file_path}")
    try:
        df = pd.read_csv(file_path, names=['Comment_ID', 'Nickname', 'User', 'User_URL', 
                                          'Comment_Text', 'Time', 'Likes', 
                                          'Profile_Picture_URL', 'Replies_Count'])
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False

    # Better data cleaning
    df['Clean_Comment'] = df['Comment_Text'].apply(clean_comment)
    
    # Remove empty comments
    df = df[df['Clean_Comment'].str.len() > 0]
    
    # Deduplicate comments
    df = df.drop_duplicates(subset=['Clean_Comment'])
    
    # Create documents for the vector store
    documents = []
    for _, row in df.iterrows():
        # Use Clean_Comment as the main content
        content = row['Clean_Comment']
        
        # Include useful metadata
        metadata = {
            'comment_id': row['Comment_ID'],
            'nickname': row['Nickname'],
            'user': row['User'],
            'time': row['Time'],
            'likes': row['Likes'],
            'replies_count': row['Replies_Count']
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    # If the database directory exists, remove it to avoid dimension mismatch errors
    if os.path.exists("./chroma_db"):
        print("Removing existing database to avoid dimension conflicts...")
        shutil.rmtree("./chroma_db")
    
    # Use all-MiniLM-L6-v2 which has 384 dimensions and works well for semantic search
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store with chunking configuration
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    vectordb.persist()
    
    print(f"Database setup complete with {len(documents)} comments!")
    return True

# Function to initialize QA system with improved retrieval
def initialize_qa_system():
    if not os.path.exists("./chroma_db"):
        print("Vector database not found. Please run the setup first.")
        return None
    
    print("Loading existing vector database...")
    # Must use the same embedding model as during setup
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Get the base retriever with MMR search for better semantic retrieval
    base_retriever = vectordb.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for diversity
        search_kwargs={
            "k": 15,  # Retrieve more documents initially
            "fetch_k": 20,  # Consider more documents for diversity
            "lambda_mult": 0.8  # Balance between relevance and diversity
        }
    )
    
    # Initialize the LLM
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=api_key,
        model_kwargs={"temperature": 0.3, "max_length": 768}  # Lower temperature for more focused answers
    )
    
    # Create a QA chain with the improved retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=base_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

# Function to ask questions to the QA system with improved response handling
def ask_question(qa_chain, question):
    try:
        print(f"\nProcessing question: {question}")
        print("Retrieving relevant comments...")
        
        result = qa_chain({"query": question})
        
        # Post-process to deduplicate source documents
        deduplicated_docs = deduplicate_documents(result['source_documents'])
        
        # Sort documents by likes count (if available) to prioritize popular comments
        try:
            deduplicated_docs = sorted(deduplicated_docs, 
                                       key=lambda x: int(x.metadata.get('likes', 0)), 
                                       reverse=True)
        except:
            pass  # If sorting fails, use the original order
        
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['result']}")
        print(f"\nSource Comments (sorted by popularity):")
        for i, doc in enumerate(deduplicated_docs):
            likes = doc.metadata.get('likes', 'Unknown')
            print(f"{i+1}. {doc.page_content}")
            print(f"   Likes: {likes} | User: {doc.metadata.get('nickname', 'Anonymous')}\n")
        
        return result
    except Exception as e:
        print(f"Error processing question: {e}")
        return None

# Main execution
if __name__ == "__main__":
    # Choose the mode
    while True:    
        mode = input("Choose mode ('setup' to build database, 'query' to ask questions, 'exit' to quit): ").strip().lower()
        
        if mode == "exit":
            break
        elif mode == "setup":
            setup_database(file_path)
        elif mode == "query":
            qa_chain = initialize_qa_system()
            if qa_chain:
                print("QA system initialized successfully.")
                while True:
                    question = input("\nEnter your question about celebrity rumors (or 'quit' to exit): ")
                    if question.lower() == 'quit':
                        break
                    ask_question(qa_chain, question)
            else:
                print("Failed to initialize QA system.")
        else:
            print("Invalid mode. Please choose 'setup', 'query', or 'exit'.")