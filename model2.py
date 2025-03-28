import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Define constants for model names
OLLAMA_MODEL_NAME = "llama3.1"
OLLAMA_SERVER_URL = "https://c967-34-169-7-63.ngrok-free.app/"
print(OLLAMA_SERVER_URL)

# Define a function to initialize the LLM based on the provider
def initialize_llm(provider, model_name, temperature=0.7, max_tokens=None, timeout=None, max_retries=2):
    """
    Initialize the LLM based on the provider and model name.
    
    Args:
        provider (str): The provider to use ("openai", "openrouter", or "groq").
        model_name (str): The name of the model to use.
        temperature (float): The temperature for sampling.
        max_tokens (int): The maximum number of tokens to generate.
        timeout (int): The timeout for the API call.
        max_retries (int): The maximum number of retries for the API call.
    
    Returns:
        The initialized LLM.
    """
    if provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        return ChatOpenAI(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif provider == "openrouter":
        if "OPENROUTER_API_KEY" not in os.environ:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
        return ChatOpenAI(
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif provider == "groq":
        if "GROQ_API_KEY" not in os.environ:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# Example usage
def generate(prompt, provider="openai", model_name="gpt-3.5-turbo", temperature=0.7):
    """
    Generate a response using the specified provider and model.
    
    Args:
        prompt (str): The input prompt.
        provider (str): The provider to use ("openai", "openrouter", or "groq").
        model_name (str): The name of the model to use.
        temperature (float): The temperature for sampling.
    
    Returns:
        The generated response.
    """
    # Initialize the LLM
    llm = initialize_llm(provider, model_name, temperature=temperature)
    
    # Generate the response
    messages = [("human", prompt),]
    ai_msg = llm.invoke(messages)
    print("AI msg: ", ai_msg)

    return ai_msg.content

# Example usage
if __name__ == "__main__":
    # Set the provider and model name
    provider = "openai"  # Change to "openrouter" or "groq" as needed
    model_name = "gpt-4o"  # Change to the desired model name
    
    # Generate a response
    prompt = "Explain the difference between INNER JOIN and LEFT JOIN in SQL."
    response = generate(prompt, provider=provider, model_name=model_name)
    print("Response:", response)