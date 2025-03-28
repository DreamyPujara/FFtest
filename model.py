# import os
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI


# import os
# if "GROQ_API_KEY" not in os.environ:
#     os.environ["GROQ_API_KEY"] = "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"



# load_dotenv()

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

# OR_llm = ChatOpenAI(
#   openai_api_key="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",
#   openai_api_base="https://openrouter.ai/api/v1",
#   model_name="qwen/qwen2.5-vl-72b-instruct:free",
#   temperature=0.7,
# )
# #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
# #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
# llm = OR_llm





def generate(prompt):
    messages = [("human", prompt),]
    ai_msg = llm.invoke(messages)
    print("AI msg: ", ai_msg)

    return ai_msg.content
