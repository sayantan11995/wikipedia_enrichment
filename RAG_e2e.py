import torch
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from pathlib import Path
import langchain
import json
import chromadb
from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

import utils


with open("config.json", "r") as content:
    config = json.load(content)

with open("book_config.json", "r") as content:
    book_config = json.load(content)

book = config.get("book")
output_json = "output_json"
dir_path = rf"books/{book}/part"
rootdir = rf"books/{book}/"
wiki_url = book_config.get(book).get("wikipedia_link")
person = book.replace("_", " ")

print(book)

Path(rf"{output_json}/{book}").mkdir(parents=True, exist_ok=True)

loader = DirectoryLoader(rootdir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
data=loader.load()

#######################
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(data)
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
#######################

vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

model_4bit = AutoModelForCausalLM.from_pretrained( model_id, device_map="auto",quantization_config=quantization_config, )
tokenizer = AutoTokenizer.from_pretrained(model_id)


pipe = pipeline(
        "text-generation",
        model=model_4bit,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=2000,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)
llm = HuggingFacePipeline(pipeline=pipe)

# Retriving Top n Chunks most Similar to query.
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

#######################
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    # chain_type="map_reduce",
    retriever=retriever,
    verbose=True
)

def run_my_rag(qa, query):
    # print(f"Query: {query}\n")
    result = qa.run(query)
    # print("\nResult: ", result)
    return result

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

wikipedia_content = utils.get_wikipedia_content(wiki_url)

generated_content = {}
for section_name, section_content in wikipedia_content.items():

    query_stage_1 = f"""You are an AI assistant in writing Wikipedia articles on personalities and your task is to expand the existing content of the given Wikipedia section about the personality: "{person}" from the source documents. Can you add 3-4 most relevant sentences to the existing content? DO NOT use any external information.
    
    Existing content: "{section_content}"
    
    New relevant sentences: """

    result = qa({"query": query_stage_1})


    ## Using the retrieved document as context to query the LLM 
    context = format_docs(result.get("source_documents", []))
    query_stage_2 = f"""You are an AI assistant in writing Wikipedia articles on personalities and your task is to expand the existing content of the given Wikipedia section about the personality: "{person}" from the given context. Using the context generate a coherent, insightful and neutral expansion of the existing content. STRCTLY Do not generate more than 4 sentences. If it is not possible to expand the content from the context, say so.

    context: "{context}"

    Existing content: "{section_name}: {section_content}"

    Expanded content: """

    generated_content[section_name] = llm(query_stage_2)

    # print("="*50)
    # print(section_content)
    # print("*"*25)
    # print(llm(query_stage_2))


old_wikipedia_content = " ".join(wikipedia_content.values())


text = ""
for section_name, text in generated_content.items():
    print(text)
    print("="*20)
    if "not possible" not in text.lower():
        wikipedia_content[section_name] += text

with open(f"{output_json}/{book}/e2e_RAG_generated_content.json", "w") as content:
    json.dump(generated_content, content)

updated_wikipedia_content = " ".join(wikipedia_content.values())


old_score = utils.calculate_quality(old_wikipedia_content)
updated_score = utils.calculate_quality(updated_wikipedia_content)

# Calculate the difference between corresponding values
difference_dict = {key: updated_score[key] - old_score[key] for key in old_score}

print(difference_dict)
