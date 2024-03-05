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
import collections


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


# Opening JSON file
f = open(rf"{output_json}/{book}/mp_bert.json", 'r')
# returns JSON object as a dictionary
mp_bert = json.load(f)

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

# Retriving Top 3 Chunks most Similar to KeyWord.
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

#######################
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    # chain_type="map_reduce",
    retriever=retriever,
    verbose=True
)


# target_keywords =[]
# for key in keyword_dict.keys():
#   for pair in keyword_dict[key]:
#     if pair[1]>=0.40:
#       target_keywords.append(pair[0])

# print(len(target_keywords))

mp_combined = {}
# for k in mp_rakun.keys():
#     k1 = set([v[0].lower() for v in mp_bert[k]])
#     k2 = set([v[0].lower() for v in mp_yake[k]])
#     k3 = set([v[0].lower() for v in mp_rakun[k]])
#     mp_combined[k] = k1.union(k2,k3)
for k in mp_bert.keys():
    k1 = set([v[0].lower() for v in mp_bert[k][:5]]) # Taking top 5
    mp_combined[k] = k1


# mp_combined
for k,v in mp_combined.items():
    print(len(v))


keyword_to_chap = collections.defaultdict(list)
for key,list_val in mp_combined.items():
    for val in list_val:
        keyword_to_chap[val].append(key)


final_keywords = set()
all_keywords = list(mp_combined.values())
for i in all_keywords:
    for val in i:
          final_keywords.add(val)
# print(keyword_to_chap)

## Creating index for keywords
keywords = {}
for idx, keyword in enumerate(keyword_to_chap.keys()):
    keywords[idx] = keyword

with open(f"{output_json}/{book}/keywords.json", "w") as content:
    json.dump(keywords, content)



sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")


langchain.debug=False

scored_mp_key_doc = {}
paragraphs = {}

person = book.replace("_", " ")
for idx, keyword in keywords.items():
    print(keyword)
    # query = f"You are an expert in writing Wikipedia articles on personalities from the information given to you about them and you want to write a Wikipedia article about {person}. Write a paragraph of at least 3-4 lines related to the keyword {keyword} which will give more information about {person}. DO NOT use any external information."

    # # Assuming run_my_rag is a function that generates the paragraph
    # generated_paragraph = utils.run_rag(qa, query)

    query_stage_1 = f"""You are an AI assistant in writing Wikipedia articles on personalities and your task is to write the most relevant and a coherent brief paragraph about the given keyphrase. Write a brief paragraph of 3-4 lines related to the keyword {keyword} which will give more information about {person}.  DO NOT use any external information.
    
    Keyword: "{keyword}"
    
    Related paragraph: """

    result = qa({"query": query_stage_1})


    ## Using the retrieved document as context to query the LLM 
    context = utils.format_docs(result.get("source_documents", []))
    query_stage_2 = f"""You are an AI assistant in writing Wikipedia articles on personalities and your task is to write the most relevant and a coherent brief paragraph about the given keyphrase. Write a brief paragraph of 3-4 lines related to the keyword {keyword} which will give more information about {person}.  DO NOT use any external information.
    
    context: "{context}"

    Keyword: "{keyword}"
    
    Related paragraph: """

    generated_paragraph = llm(query_stage_2)

    print(generated_paragraph)
    kw_para_sim_score = utils.calculate_similarity(sentence_transformer_model, generated_paragraph, keyword)
    print(kw_para_sim_score)

    scored_mp_key_doc[idx] = [generated_paragraph, kw_para_sim_score, idx]
    paragraphs[idx] = generated_paragraph

with open(rf"{output_json}/{book}/scored_mp_key_doc.json", "w") as outfile:
    json.dump(scored_mp_key_doc, outfile)

with open(f"{output_json}/{book}/paragraphs.json", "w") as content:
    json.dump(paragraphs, content)