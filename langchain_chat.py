from langchain_openai.chat_models import ChatOpenAI
from config import OPENAI_API
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API


def ask_question(question: str):
    """Process user questions and return answers using LangChain."""

    # Initialize the OpenAI LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)

    # Generate the answer
    try:
        response = llm.invoke(question)
        return response
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    print("Welcome to the LangChain Question-Answering App!")
    print("Type 'exit' to quit the application.\n")

    while True:
        user_input = input("You : ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        answer = ask_question(user_input)
        content = getattr(answer, 'content', 'No content found')
        print(f"Answer: {content}")
