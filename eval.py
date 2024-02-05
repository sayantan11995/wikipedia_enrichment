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


with open("config.json", "r") as content:
    config = json.load(content)

with open("book_config.json", "r") as content:
    book_config = json.load(content)

book = config.get("book")
output_json = "output_json"
dir_path = rf"books/{book}/part"
rootdir = rf"books/{book}/"
url = book_config.get(book).get("wikipedia_link")



with open(f"{output_json}/{book}/paragraphs.json", "r") as content:
  keyword_paragraph_map = json.load(content)

with open(f"{output_json}/{book}/top_section_keywords_mapping.pkl", "rb") as content:
    top_section_keywords_mapping = pickle.load(content)

print(top_section_keywords_mapping)

# sys.exit(0)
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


old_wikipedia_content = " ".join(section_id_to_section_content.values())


text = ""
for items in top_section_keywords_mapping:
    text = keyword_paragraph_map[str(items[1])]
    section_id = str(items[0])

    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("”", "'")
    text = text.replace("“", "'")
    section_id_to_section_content[section_id] += text

    print("="*50)
    print(str(items[2]))
    print(text)

updated_wikipedia_content = " ".join(section_id_to_section_content.values())


old_score = utils.calculate_quality(old_wikipedia_content)
updated_score = utils.calculate_quality(updated_wikipedia_content)

# Calculate the difference between corresponding values
difference_dict = {key: updated_score[key] - old_score[key] for key in old_score}

print(difference_dict)
