import os
from langchain.llms import LlamaCpp
from transformers import LlamaForCausalLM, LlamaTokenizer
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, PromptHelper

# Set up the LlamaCpp model
llm = LlamaCpp(
    model_path="./dalai/alpaca/models/7B/ggml-model-q4_0.bin", verbose=True
)

llm_predictor = LLMPredictor(llm=llm)

max_input_size = 240
num_output = 120
max_chunk_overlap = 0
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# Load the documents and build the index
documents = SimpleDirectoryReader('./plugins/').load_data()

# Create a ServiceContext instance with the custom tokenizer
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

# Query the index and print the response
query_engine = index.as_query_engine()
response = query_engine.query("how do i open the instructions file?")
print(response)
