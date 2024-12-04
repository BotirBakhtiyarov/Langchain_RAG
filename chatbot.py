import os
from langchain_community.document_loaders import TextLoader  # Updated loader for text files
from langchain_openai.embeddings import OpenAIEmbeddings  # Updated embeddings
from langchain_chroma.vectorstores import Chroma  # Updated vector store
from langchain.chains import RetrievalQA  # Retrieval-Augmented Generation (RAG)
from langchain_openai.chat_models import ChatOpenAI  # Updated Chat model for OpenAI
from config import OPENAI_API  # Replace with your actual OpenAI API configuration

# 1. Configure OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API


# 2. Function to load knowledge base and preprocess data
def load_knowledge_base(folder_path):
    """
    Load text files from a specified folder and create a ChromaDB vector store.
    """
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):  # Only process .txt files
            loader = TextLoader(os.path.join(folder_path, file_name), encoding="utf-8")
            documents.extend(loader.load())  # Load documents

    # Create embeddings and initialize vector store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents, embeddings, persist_directory="./chroma_store"
    )
    return vector_store


# 3. Function to create a RAG chain
def create_rag_chain(vector_store):
    """
    Create a Retrieval-Augmented Generation (RAG) chain.
    """
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model="gpt-4", temperature=0.9)  # Using GPT-4 for stability
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


# 4. Function to update the knowledge base
def update_knowledge_base(vector_store, file_path):
    """
    Update the knowledge base by adding a new file to the vector store.
    """
    try:
        if file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            new_documents = loader.load()

            # Add new documents to the vector store
            vector_store.add_documents(new_documents)
            print(f"Knowledge base updated with file: {file_path}")
        else:
            print("Error: Only .txt files are supported.")
    except Exception as e:
        print(f"Error updating knowledge base: {e}")


# 5. Interactive chatbot function
def interactive_chatbot(qa_chain, retriever, vector_store):
    """
    Interactive chatbot with automatic mode switching and dynamic database updates.
    """
    print("Welcome to the Intelligent Chatbot!")
    print("Type 'exit' to end the session, or 'upload <file_path>' to update the knowledge base.")


    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        elif user_input.lower().startswith("upload"):
            try:
                _, file_path = user_input.split(" ", 1)
                if os.path.exists(file_path):
                    update_knowledge_base(vector_store, file_path)
                    retriever = vector_store.as_retriever()  # Refresh retriever
                else:
                    print("Error: File path does not exist.")

            except Exception as e:
                print(f"Error processing upload command: {e}")

        else:
            try:
                # Attempt knowledge base retrieval
                response = qa_chain.invoke({"query": user_input})["result"]
                print(f"Chatbot (Knowledge Base): {response}")


            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    # Folder containing knowledge base files
    knowledge_base_folder = "./data"

    # Step 1: Load or initialize the vector database
    if not os.path.exists("./chroma_store"):  # If no saved database exists
        vector_store = load_knowledge_base(knowledge_base_folder)
    else:
        # Load existing vector store
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma(persist_directory="./chroma_store", embedding_function=embeddings)

    # Step 2: Create RAG chain
    qa_chain = create_rag_chain(vector_store)
    retriever = vector_store.as_retriever()

    # Step 3: Launch the chatbot
    interactive_chatbot(qa_chain, retriever, vector_store)
