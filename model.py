import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


# import os
# if "GROQ_API_KEY" not in os.environ:
#     os.environ["GROQ_API_KEY"] = "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"



load_dotenv()

# # OLLAMA_MODEL_NAME = "llama3.1"
# # OLLAMA_SERVER_URL = "https://c967-34-169-7-63.ngrok-free.app/"
# # print(OLLAMA_SERVER_URL)




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
  model_name="meta-llama/llama-3.3-70b-instruct:free",
)
#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
llm = OR_llm





def generate(prompt):
    messages = [("human", prompt),]
    ai_msg = llm.invoke(messages)
    print("AI msg: ", ai_msg)

    return ai_msg.content
