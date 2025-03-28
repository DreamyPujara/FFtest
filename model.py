import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import streamlit as st


# import os
# if "GROQ_API_KEY" not in os.environ:
#     os.environ["GROQ_API_KEY"] = "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"
# #client = Groq(api_key=st.secrets["GROQ_API_KEY"])


load_dotenv()

OLLAMA_MODEL_NAME = "llama3.1"
OLLAMA_SERVER_URL = "https://c967-34-169-7-63.ngrok-free.app/"
print(OLLAMA_SERVER_URL)




# G_llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )
# llm=G_llm

OR_llm = ChatOpenAI(
  openai_api_key="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",
  openai_api_base="https://openrouter.ai/api/v1",
  model_name="deepseek/deepseek-r1-distill-llama-70b:free",
  temperature=0.7,
  headers = {
            "HTTP-Referer": st.secrets.get("APP_URL", "https://fftest.streamlit.app/"),  # Your app URL
            "Authorization": f"Bearer {"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"}"
        }
  
)
#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
#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
llm = OR_llm





def generate(prompt):
    messages = [("human", prompt),]
    ai_msg = llm.invoke(messages)
    print("AI msg: ", ai_msg)

    return ai_msg.content
