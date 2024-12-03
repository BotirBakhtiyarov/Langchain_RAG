# Intelligent Chatbot with Retrieval-Augmented Generation (RAG)

This project implements an intelligent chatbot that combines a **knowledge base** with OpenAI's GPT-4 for Retrieval-Augmented Generation (RAG). The chatbot can answer questions based on your custom knowledge base or fallback to general conversational abilities using ChatGPT. 

---

## Features

- **Knowledge Base Integration**: Automatically retrieves relevant documents from a local knowledge base and generates context-specific answers.
- **General ChatGPT Fallback**: Switches to general conversational mode if no relevant knowledge base content is found.
- **Interactive Chat**: Provides a user-friendly interface for real-time interaction.
- **Persistence**: Saves and loads the vector database for efficient and reusable embeddings.

---

## Requirements

Before using this project, ensure you have the following installed:

- Python 3.8 or above
- Required Python libraries (install via `requirements.txt`)

---

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/BotirBakhtiyarov/Langchain_RAG.git
   cd Langchain_RAG
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key**:
   - Add your OpenAI API key in a `config.py` file as follows:
     ```python
     OPENAI_API = "your-openai-api-key"
     ```

4. **Prepare the knowledge base**:
   - Create a folder named `data/` in the root directory.
   - Add `.txt` files with the knowledge base content inside the `data/` folder.

5. **Run the chatbot**:
   ```bash
   python main.py
   ```

---

## Usage

1. Start the chatbot:
   ```bash
   python main.py
   ```

2. Interact with the chatbot:
   - Type your queries.
   - Type `exit` to end the session.

---

## Project Structure

```
intelligent-chatbot/
├── chroma_store/          # Persisted vector store (auto-generated)
├── data/                  # Folder for your knowledge base (.txt files)
├── main.py                # Main script for the chatbot
├── config.py              # Configuration for API keys
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Dependencies

Install the required dependencies using the following:
```bash
pip install langchain chromadb openai
```

---

## Example Interaction

```plaintext
Welcome to the Intelligent Chatbot! Type 'exit' to end the session.

You: What is the capital of France?
Chatbot (General ChatGPT): The capital of France is Paris.

You: What is in the company handbook?
Chatbot (Knowledge Base): The company handbook explains policies on working hours, leave, and conduct. Please refer to section 2.1 for details.

You: exit
Chatbot: Goodbye!
```

---

## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to submit a pull request or open an issue.

---

Enjoy your intelligent chatbot! 😊
