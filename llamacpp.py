from llama_cpp import Llama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
      model_path="/home/aleks/projects/turbo-genius/models/Mistral-7B-Instruct-v0.3.Q6_K.gguf",
                max_tokens =1000,
                n_threads = 8,
                temperature= 0.8,
                n_gpu_layers=24,
                verbose=True,
                top_p=0.75,
                top_k=40,
        )

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate.from_template(template)


llm_chain = prompt | llm
question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
for chunk in llm_chain.stream({"question": question}):
    print(chunk, end="", flush=True)