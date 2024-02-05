import utils

import re
import os, os.path
import fnmatch
import json
import collections
import glob
import json

import matplotlib.pyplot as plt
import numpy as np
# import modules for web scrapping
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util
import torch
import math
import pickle


with open("config.json", "r") as content:
    config = json.load(content)

with open("book_config.json", "r") as content:
    book_config = json.load(content)

book = config.get("book")
output_json = "output_json"
dir_path = rf"books/{book}/part"
rootdir = rf"books/{book}/"


## HYPERPARMETERS
alpha = 1
beta = -1
gamma = 1


url = book_config.get(book).get("wikipedia_link")
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
response = requests.get(url, headers=headers)
#print(response.status_code)
soup = BeautifulSoup(response.content, 'html.parser')

wikipedia_section_names = []
non_relevant_sections = ["See_also", "References", "Footnotes", "External_links", "Notes", "Bibliography", "Further_reading"]
for link in soup.find_all('span', attrs={'class':'mw-headline'}):
    if link.get('id') is not None and link.get('id') not in non_relevant_sections:
        wikipedia_section_names.append(link.get('id'))
print(wikipedia_section_names)

def FetchParagraphBetweenIds(id1,id2):
    hElem = soup.find("span", {'id': id1})
    endElem = soup.find('span', {'id': id2})
    cntns = list(soup.find_all())

    my_lst = []
    inBetween = False
    for tag in cntns:
        if tag == hElem:
            inBetween = True
        if inBetween == True and tag.name == 'p':
            my_lst.append(tag.get_text())
        if tag == endElem:
            inBetween = False
            break
    return "".join(my_lst)

section_id_to_section_content = {}
for i in range(len(wikipedia_section_names)-1):
    section_id_to_section_content[wikipedia_section_names[i]] = FetchParagraphBetweenIds(wikipedia_section_names[i],wikipedia_section_names[i+1])



mp_sim_score = {}
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embedder.to("cuda")


with open(rf"{output_json}/{book}/scored_mp_key_doc.json", "r") as outfile:
    scored_mp_key_doc = json.load(outfile)

with open(rf"{output_json}/{book}/keywords.json", "r") as outfile:
    keywords = json.load(outfile)

mp_section_paragraph = {}
mp_section_keyword = {}
for section_id, section_content in section_id_to_section_content.items():
    print("Processing section: ", section_id)
    for key_idx, paragraph_and_score in scored_mp_key_doc.items():
        paragraph = paragraph_and_score[0]
        if section_content != '' and paragraph != '':
            section_embeddings = embedder.encode(section_content, convert_to_tensor=True)
            paragraph_embedding = embedder.encode(paragraph, convert_to_tensor=True)
            keyword_embedding = embedder.encode(keywords[key_idx], convert_to_tensor=True)

            section_para_similarity = util.cos_sim(section_embeddings, paragraph_embedding)[0][0].detach().cpu().numpy()
            section_keyword_similarity = util.cos_sim(section_embeddings, keyword_embedding)[0][0].detach().cpu().numpy()
            # mp_sim_score[str(key1)+'->'+str(key2)] = cos_scores
            paragraph_keyword_similarity = paragraph_and_score[1]

            # section_para_similarity = section_para_similarity.detach().cpu().numpy()

            weighted_score = (alpha * paragraph_keyword_similarity) + (beta * section_para_similarity) + (gamma * section_keyword_similarity)

            # Storing Keyword_idx, keyword, score
            ## sec->para * para -> keyword
            if str(section_id) in mp_section_keyword.keys():
                mp_section_keyword[str(section_id)].append([key_idx, str(keywords[key_idx]), weighted_score])
            else:
                mp_section_keyword[str(section_id)] = [[key_idx, str(keywords[key_idx]), weighted_score]]



mp_section_keyword
sorted_mp_section_keyword = {}

for key, values in mp_section_keyword.items():
    # print(values)
    sorted_list = sorted(values, key=lambda x: x[2], reverse=True)[:6]
    sorted_mp_section_keyword[key] = sorted_list

## Finding top 5 keywords
topn = 15
# Create an empty list to store inner keys and values along with their parent keys
all_inner_keys = []

# Iterate through the dictionary and extract inner keys, values, and parent keys
for parent_key, inner_list in sorted_mp_section_keyword.items():
    for key_idx, inner_key, value in inner_list:
        all_inner_keys.append((parent_key, key_idx, inner_key, value))

# Sort the list of inner keys based on their values in descending order
all_inner_keys.sort(key=lambda x: x[3], reverse=True)

# Extract the top 3 inner keys with highest values along with their parent keys
top_inner_keys = [(parent_key, key_idx, inner_key, value) for parent_key, key_idx, inner_key, value in all_inner_keys[:topn]]

print(top_inner_keys)

with open(f"{output_json}/{book}/top_section_keywords_mapping.pkl", "wb") as content:
    pickle.dump(top_inner_keys, content)

