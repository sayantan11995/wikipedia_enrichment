import torch
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from pathlib import Path
import langchain
import json
import chromadb
import pickle
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

from sgnlp.models.coherence_momentum import CoherenceMomentumModel, CoherenceMomentumConfig, \
    CoherenceMomentumPreprocessor

# Load Model
model_path = "./CoherenceMomentum"
config = CoherenceMomentumConfig.from_pretrained(
    model_path
)
# coherence_model = CoherenceMomentumModel.from_pretrained(
#     "CoherenceMomentumModel/pytorch_model.bin",
#     config=config
# )
coherence_model = CoherenceMomentumModel.from_pretrained(model_path)
coherence_model.to("cuda")

preprocessor = CoherenceMomentumPreprocessor(config.model_size, config.max_len)


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

## Load wikipedia content
wikipedia_content = utils.get_wikipedia_content(wiki_url)

## From the kw based approach get the top n wikipedia sections (pre-calculated)
with open(f"{output_json}/{book}/top_section_keywords_mapping.pkl", "rb") as content:
    top_section_keywords_mapping = pickle.load(content) ## contain [(section_title, kw_id, kw, score),...]

# top_section_keywords_mapping = [('Private_life', '23', 'adams', 0.8977261781692505), ('Career', '23', 'adams', 0.7279162406921387), ('A_Memoir_of_Miss_Hannah_Adams', '23', 'adams', 0.6677132248878479), ('A_Summary_History_of_New-England', '23', 'adams', 0.6353128254413605), ('A_View_of_Religions', '23', 'adams', 0.5861822366714478), ('Early_years_and_education', '23', 'adams', 0.44095492362976074), ('A_View_of_Religions', '21', 'iii', 0.4263700991868973), ('History_of_the_Jews_and_Letters_on_the_Gospels', '23', 'adams', 0.36805176734924316), ('A_View_of_Religions', '7', 'books', 0.36610615253448486), ('History_of_the_Jews_and_Letters_on_the_Gospels', '7', 'books', 0.3542388677597046), ('Career', '16', 'interests', 0.3316535949707031), ('A_Summary_History_of_New-England', '7', 'books', 0.31932157278060913), ('Career', '7', 'books', 0.30555248260498047), ('A_View_of_Religions', '16', 'interests', 0.2919265776872635), ('A_Memoir_of_Miss_Hannah_Adams', '7', 'books', 0.2854318767786026)]

selected_sections = list(set([items[0] for items in top_section_keywords_mapping]))
print(selected_sections)

## Load e2e RAG based generated content
with open(f"{output_json}/{book}/e2e_RAG_generated_content.json", "r") as content:
    e2e_RAG_generated_content = json.load(content)

with open(rf"{output_json}/{book}/scored_mp_key_doc.json", "r") as outfile:
    scored_mp_key_doc = json.load(outfile)


old_wikipedia_content = " ".join(wikipedia_content.values())

for section_name, kw_id, _, _ in top_section_keywords_mapping[:5]:
    updated_text_from_e2e = wikipedia_content[section_name] + e2e_RAG_generated_content[section_name]
    updated_text_from_kw_based = wikipedia_content[section_name] + scored_mp_key_doc[kw_id][0]
    

    updated_text_from_e2e_tensor = preprocessor([updated_text_from_e2e])
    updated_text_from_e2e_coherence_score = coherence_model.get_main_score(updated_text_from_e2e_tensor["tokenized_texts"].to("cuda")).item()

    updated_text_from_kw_based_tensor = preprocessor([updated_text_from_kw_based])
    updated_text_from_kw_based_coherence_score = coherence_model.get_main_score(updated_text_from_kw_based_tensor["tokenized_texts"].to("cuda")).item()

    ## Check coherence score
    if updated_text_from_e2e_coherence_score > updated_text_from_kw_based_coherence_score:
        print("#"* 20 + "E2E")
        wikipedia_content[section_name] +=  updated_text_from_e2e
    else:
        wikipedia_content[section_name] +=  updated_text_from_kw_based
        print("#"* 20 + "Keyword")

updated_wikipedia_content = " ".join(wikipedia_content.values())


old_score = utils.calculate_quality(old_wikipedia_content)
updated_score = utils.calculate_quality(updated_wikipedia_content)

# Calculate the difference between corresponding values
difference_dict = {key: updated_score[key] - old_score[key] for key in old_score}

print(difference_dict)
