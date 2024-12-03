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
    vector_store.persist()  # Persist the database
    return vector_store


# 3. Function to create a RAG chain
def create_rag_chain(vector_store):
    """
    Create a Retrieval-Augmented Generation (RAG) chain.
    """
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model="gpt-4", temperature=0.9)  # Using GPT-4 for stability
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


# 4. General ChatGPT response
def ordinary_chat(user_input):
    """
    Respond using general ChatGPT (not the knowledge base).
    """
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0.9)
        response = llm([{"role": "user", "content": user_input}])
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error in general chat response: {e}"


# 5. Interactive chatbot function
def interactive_chatbot(qa_chain, retriever):
    """
    Interactive chatbot with automatic mode switching.
    """
    print("Welcome to the Intelligent Chatbot! Type 'exit' to end the session.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        try:
            # Step 1: Attempt knowledge base retrieval
            retrieved_docs = retriever.get_relevant_documents(user_input)  # Retrieve relevant documents
            if retrieved_docs:  # If relevant documents are found
                response = qa_chain({"query": user_input})["result"]  # Properly extract the response
                print(f"Chatbot (Knowledge Base): {response}")
            else:
                # Step 2: Fallback to general ChatGPT
                response = ordinary_chat(user_input)
                print(f"Chatbot (General ChatGPT): {response}")
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
    interactive_chatbot(qa_chain, retriever)
