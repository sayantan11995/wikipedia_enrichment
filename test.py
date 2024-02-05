# from sgnlp.models.coherence_momentum import CoherenceMomentumModel, CoherenceMomentumConfig, \
#     CoherenceMomentumPreprocessor

# # Load Model
# config = CoherenceMomentumConfig.from_pretrained(
#     "https://storage.googleapis.com/sgnlp-models/models/coherence_momentum/config.json"
# )
# model = CoherenceMomentumModel.from_pretrained(
#     "https://storage.googleapis.com/sgnlp-models/models/coherence_momentum/pytorch_model.bin",
#     config=config
# )

# preprocessor = CoherenceMomentumPreprocessor(config.model_size, config.max_len)

# # Example text inputs
# text1 = "Companies listed below reported quarterly profit substantially different from the average of analysts ' " \
#         "estimates . The companies are followed by at least three analysts , and had a minimum five-cent change in " \
#         "actual earnings per share . Estimated and actual results involving losses are omitted . The percent " \
#         "difference compares actual profit with the 30-day estimate where at least three analysts have issues " \
#         "forecasts in the past 30 days . Otherwise , actual profit is compared with the 300-day estimate . " \
#         "Source : Zacks Investment Research"
# text2 = "Companies listed below reported quarterly profit substantially different from the average of analysts ' " \
#         "estimates . The companies are followed by at least three analysts , and had a minimum five-cent change in " \
#         "actual earnings per share . Estimated and actual results involving losses are omitted . The percent " \
#         "difference compares actual profit with the 30-day estimate where at least three analysts have issues " \
#         "forecasts in the past 30 days . Otherwise , actual profit is compared with the 300-day estimate . " \
#         "Source : Zacks Investment Research"\
#         "Companies listed below reported quarterly profit substantially different from the average of analysts ' " \
#         "estimates . The companies are followed by at least three analysts , and had a minimum five-cent change in " \
#         "actual earnings per share . Estimated and actual results involving losses are omitted . The percent " \
#         "difference compares actual profit with the 30-day estimate where at least three analysts have issues " \
#         "forecasts in the past 30 days . Otherwise , actual profit is compared with the 300-day estimate . " \
#         "Source : Zacks Investment Research"

# text1_tensor = preprocessor([text1])
# text2_tensor = preprocessor([text2])

# text1_score = model.get_main_score(text1_tensor["tokenized_texts"]).item()
# text2_score = model.get_main_score(text2_tensor["tokenized_texts"]).item()

# print(text1_score, text2_score)


import pickle
import utils

import re
import os, os.path, sys
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
from sgnlp.models.coherence_momentum import CoherenceMomentumModel, CoherenceMomentumConfig, \
    CoherenceMomentumPreprocessor


# def parse_input_text(input_text):
#     # Split the input text into lines
#     lines = input_text.strip().split('\n')

#     # Initialize an empty dictionary to store keyword-paragraph pairs
#     keyword_paragraph_dict = {}

#     # Process each line to extract keyword and paragraph
#     current_keyword = None
#     for line in lines:
#         # Split each line into key and value
#         key, value = line.split(': ', 1)

#         # Check if it's a keyword or paragraph
#         if key.lower() == 'keyword':
#             current_keyword = value.strip()
#         elif key.lower() == 'paragraph' and current_keyword:
#             keyword_paragraph_dict[current_keyword] = value.strip()

#     return keyword_paragraph_dict


# with open("key_para.txt", "r") as content:
#     text = content.read()

# keyword_paragraph_dict = parse_input_text(text)

# print(keyword_paragraph_dict)
# sys.exit(0)

# Load Model
model_path = "./CoherenceMomentumModel"
config = CoherenceMomentumConfig.from_pretrained(
    "CoherenceMomentumModel/config.json"
)
coherence_model = CoherenceMomentumModel.from_pretrained(
    "CoherenceMomentumModel/pytorch_model.bin",
    config=config
)
# coherence_model = CoherenceMomentumModel.from_pretrained(model_path)
coherence_model.to("cuda")

preprocessor = CoherenceMomentumPreprocessor(config.model_size, config.max_len)


with open("config.json", "r") as content:
    config = json.load(content)

with open("book_config.json", "r") as content:
    book_config = json.load(content)

f = open("out.txt", "w")

section_wise_coherence_score_diff = []
book = config.get("book")
output_json = "output_json"
dir_path = rf"books/{book}/part"
rootdir = rf"books/{book}/"
# url = "https://en.wikipedia.org/wiki/John_G._B._Adams"
url = book_config.get(book).get("wikipedia_link")



with open(f"{output_json}/{book}/paragraphs.json", "r") as content:
    keyword_paragraph_map = json.load(content)

with open(f"{output_json}/{book}/top_section_keywords_mapping.pkl", "rb") as content:
    top_section_keywords_mapping = pickle.load(content)

# print(top_section_keywords_mapping)


headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
response = requests.get(url, headers=headers)
#print(response.status_code)
soup = BeautifulSoup(response.content, 'html.parser')

desired_ids = []
for link in soup.find_all('span', attrs={'class':'mw-headline'}):
    if link.get('id') is not None:
        desired_ids.append(link.get('id'))
# print(desired_ids)

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
for i in range(len(desired_ids)-1):
    section_id_to_section_content[desired_ids[i]] = FetchParagraphBetweenIds(desired_ids[i], desired_ids[i+1])


# old_wikipedia_content = " ".join(section_id_to_section_content.values())

text = ""

print(top_section_keywords_mapping)
for items in top_section_keywords_mapping:
    text = "John_G_B_Adams, also known as John Gray, was a prominent leader in the American civil war, serving in the 2nd Massachusetts Volunteer Regiment. He was a major and commanding officer of Company I, one of the four companies that made up the regiment. During his time in the regiment, Adams was responsible for ensuring the safety and well-being of his men in combat. He was known for his bravery and leadership abilities, and was respected by both his fellow officers and his soldiers. Despite the danger and uncertainty of war, Adams remained determined to see the conflict through to victory, and his dedication to his cause was evident in his actions on the battlefield."
    section_id = str(items[0])

    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("”", "'")
    text = text.replace("“", "'")

    old_text = section_id_to_section_content[section_id]
    updated_text = old_text + text

    text1_tensor = preprocessor([old_text])
    text2_tensor = preprocessor([updated_text])

    text1_score = coherence_model.get_main_score(text1_tensor["tokenized_texts"].to("cuda")).item()
    text2_score = coherence_model.get_main_score(text2_tensor["tokenized_texts"].to("cuda")).item()

    print(text1_score, text2_score)
    difference = text2_score-text1_score
    section_wise_coherence_score_diff.append(difference)

f.close()