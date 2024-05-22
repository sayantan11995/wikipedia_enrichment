### This file can be used using the model loaded by unsloth FastLanguageModel

import torch
from unsloth import FastLanguageModel
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from pathlib import Path
import langchain
import json
import shutil
import chromadb
from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from tqdm import tqdm
import glob
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

import utils


with open("config.json", "r") as content:
    config = json.load(content)

with open("book_config.json", "r") as content:
    book_config = json.load(content)


book = config.get("book")
book_path = config.get("book_path") # whether to choose tesseract or langchain OCR
output_json = "output_json"
dir_path = rf"books/{book_path}/{book}/part"
rootdir = rf"books/{book_path}/{book}/"
wiki_url = book_config.get(book).get("wikipedia_link")
person = book.replace("_", " ")

print(book)
print(rootdir)


Path(rf"{output_json}/{book}").mkdir(parents=True, exist_ok=True)

loader = DirectoryLoader(rootdir, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
data=loader.load()

#######################
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", "."],)
all_splits = text_splitter.split_documents(data)
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
#######################
print(all_splits[0])
print("()"*50)

shutil.rmtree("chroma_db", ignore_errors=True)
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

save_path = "/content/drive/MyDrive/Colab Notebooks/wikipedia_enrichment/saved_models/finetuned_llama-2-7b-wikipedia-section-completion"

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = save_path, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
As a wikipedia writing assistant, generate a coherent paragraph relevant to the given section content from the external content.

### Input:
Section Content: {}: {}

### Exteral Content:
{}

### Response:
{}"""



# from transformers import TextStreamer
# text_streamer = TextStreamer(tokenizer)

# llm = HuggingFacePipeline(pipeline=pipe)

# Retriving Top n Chunks most Similar to query.
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

#######################

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

wikipedia_content = utils.get_wikipedia_content(wiki_url)

generated_content = {}
for section_name, section_content in tqdm(wikipedia_content.items()):

    print(f"Processig section: {section_name}")

    query_stage_1 = f"""{section_name}: {section_content}"""

    # result = qa({"query": query_stage_1})


    ## Using the retrieved document as context to query the LLM 
    # context = format_docs(result.get("source_documents", []))
    context = "\n".join([doc.page_content for doc in retriever.get_relevant_documents(query_stage_1)])
    # print(f"Retrieved Context: {context}")

    query_stage_2 = alpaca_prompt.format(
            section_name,
            section_content,
            context,
            "", # output - leave this blank for generation!
        )

    inputs = tokenizer(
    [
        query_stage_2
    ], return_tensors = "pt").to("cuda")


    output = model.generate(**inputs, max_new_tokens = 128)
    output_text = tokenizer.decode(output[0], skip_special_tokens = True)
    generated_content[section_name] = output_text[len(query_stage_2):].strip()



old_wikipedia_content = " ".join(wikipedia_content.values())

print(f"Old score: {utils.calculate_quality(old_wikipedia_content)}")


text = ""
for section_name, text in generated_content.items():
    print(f"Section: {section_name}")
    print("Generated content: ")
    print(text)
    print("="*20)
    if "not possible" not in text.lower():
        wikipedia_content[section_name] += text

with open(f"{output_json}/{book}/e2e_RAG_generated_content.json", "w") as content:
    json.dump(generated_content, content)

updated_wikipedia_content = " ".join(wikipedia_content.values())


old_score = utils.calculate_quality(old_wikipedia_content)
updated_score = utils.calculate_quality(updated_wikipedia_content)
print(f"Updated score: {updated_score}")

# Calculate the difference between corresponding values
difference_dict = {key: updated_score[key] - old_score[key] for key in old_score}

print(difference_dict)
