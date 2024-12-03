from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma
import os
from config import OPENAI_API

# Configure OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API


# Function to load knowledge base and create a vector store
def load_knowledge_base(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):  # Only process .txt files
            loader = TextLoader(os.path.join(folder_path, file_name), encoding="utf-8")
            documents.extend(loader.load())

    embeddings = OpenAIEmbeddings()

    # Create the Chroma vector store and pass the persist_directory when creating it
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_store")

    # Chroma will handle persistence automatically with the directory provided.
    return vector_store


# Function to create RAG chain
def create_rag_chain(vector_store):
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model="gpt-4", temperature=1)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


def interactive_chatbot(qa_chain, retriever, general_llm):
    """
    Interactive chatbot with fallback for out-of-domain questions.
    """
    print("Welcome to the Intelligent Chatbot! Type 'exit' to end the session.")
    print("Type 'mode' to switch between Knowledge Base and General ChatGPT Mode.")

    # Initialize mode as Knowledge Base mode
    current_mode = "knowledge_base"

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Switch between modes
        elif user_input.lower() == "mode":
            if current_mode == "knowledge_base":
                current_mode = "general_chatgpt"
                print("Chatbot: Switched to General ChatGPT Mode.")
            else:
                current_mode = "knowledge_base"
                print("Chatbot: Switched to Knowledge Base Mode.")

        try:
            if current_mode == "knowledge_base":
                # Knowledge Base Mode: Retrieve relevant documents from the vector store
                retrieved_docs = retriever.invoke(user_input, n_results=5)  # Correctly pass n_results here
                if not retrieved_docs:
                    print(
                        "Chatbot: Sorry, I couldn't find a specific answer in my knowledge base. Here's a general response:")
                    response = general_llm.predict(user_input)  # Use predict() instead of call()
                    print(f"Chatbot (General Knowledge): {response}")
                else:
                    # Use retrieved documents to provide an answer
                    documents = [doc.page_content for doc in retrieved_docs]  # Extract content from docs
                    response = qa_chain.invoke({"query": user_input, "documents": documents})["result"]
                    print(f"Chatbot (Knowledge Base): {response}")

            elif current_mode == "general_chatgpt":
                # General ChatGPT Mode: Direct response from ChatGPT (out-of-domain queries)
                response = general_llm.invoke(user_input)  # Use predict() here as well
                print(f"Chatbot (General ChatGPT): {response}")

        except Exception as e:
            print(f"Error: {e}")


# Main function to start chatbot
if __name__ == "__main__":
    # Path to folder containing text files for the knowledge base
    knowledge_base_folder = "./data"

    # Load the knowledge base (only once, then use the vector store)
    if not os.path.exists("./chroma_store"):
        vector_store = load_knowledge_base(knowledge_base_folder)
    else:
        vector_store = Chroma(persist_directory="./chroma_store")
        if not vector_store._embedding_function:  # Accessing private attribute
            vector_store._embedding_function = OpenAIEmbeddings()

    # Create RAG chain using the vector store
    qa_chain = create_rag_chain(vector_store)

    # Initialize general ChatGPT model for fallback
    general_llm = ChatOpenAI(model="gpt-4", temperature=0.8)

    # Start the interactive chatbot
    interactive_chatbot(qa_chain, vector_store.as_retriever(), general_llm)
