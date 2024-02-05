import utils

import re
import os, os.path
import fnmatch
import json
import collections
import glob
import json
from pathlib import Path
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


# if not os.path.exists(rf"{output_json}/{book}"): 
#     os.makedirs(rf"{output_json}/{book}") 
Path(rf"{output_json}/{book}").mkdir(parents=True, exist_ok=True)

part = 1
mp_bert, mp_yake, mp_rakun = {}, {}, {}
for path in glob.glob(f'{rootdir}/*/**/', recursive=True):
    x = 1
    no_of_chapters = len(fnmatch.filter(os.listdir(dir_path + str(part) + '/'), '*.txt'))
    print(no_of_chapters)
    while x <= no_of_chapters:
        target_x = no_of_chapters+1
        for i in range(x,target_x):
            with open(dir_path + str(part) + '/chapter'+ str(i) + '.txt', 'r', encoding='utf-8') as content_file:
                key = str(part) + '/chapter'+ str(i)
                content = content_file.read()
                content = utils.clean_ocr_text(content)
                print(key)
                mp_bert[key] = utils.get_keyword_bert(content)
                # mp_yake[key] = utils.get_keyword_yake(content)
                # mp_rakun[key] = utils.get_keyword_rakun(content)
        x = target_x
        part += 1

with open(rf"{output_json}/{book}/mp_bert.json", "w") as outfile:
    json.dump(mp_bert, outfile)

with open(rf"{output_json}/{book}/mp_yake.json", "w") as outfile:
    json.dump(mp_yake, outfile)

with open(rf"{output_json}/{book}/mp_rakun.json", "w") as outfile:
    json.dump(mp_rakun, outfile)

mp_combined = {}
# for k in mp_rakun.keys():
#     k1 = set([v[0].lower() for v in mp_bert[k]])
#     k2 = set([v[0].lower() for v in mp_yake[k]])
#     k3 = set([v[0].lower() for v in mp_rakun[k]])
#     mp_combined[k] = k1.union(k2,k3)
for k in mp_bert.keys():
    k1 = set([v[0].lower() for v in mp_bert[k]])
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




itr = 0
mp_key_doc = {}
for key_idx, (key, val) in tqdm(enumerate(keyword_to_chap.items())):
    mp_key_doc[key_idx] = ["", [], itr] #para, score, para_index
    print(key_idx, key)
    # print(f"{key}")
    text = ""
    for k in val:
        print(k)
        with open(rf"books/{book}/part%s.txt"%k, encoding="utf8") as f:
            text += utils.clean_ocr_text(f.read())

    corpus = utils.split_sentence_from_text(text)
#   doc = create_representative_doc(corpus,key)
    para, score = utils.create_representative_doc(corpus,key)
    mp_key_doc[key_idx][0] += para
    mp_key_doc[key_idx][1].append(score)
    itr += 1
    # if itr == 100:
    #     break

## Creating keyword-paragraph index mapping, and Excluding from index if len(para) < 0

keyword_paragraph_idx_map = {}
scored_mp_key_doc = {}
para_idx = 0
for key_idx, val in mp_key_doc.items():
    para = mp_key_doc[key_idx][0]
    scores = mp_key_doc[key_idx][1]
    # para_idx = mp_key_doc[key][2]
    if len(para.strip()) >= 0:
        scored_mp_key_doc[key_idx] = [para, sum(scores)/len(scores), para_idx]
        keyword_paragraph_idx_map[key_idx] = para_idx

        para_idx += 1


print(scored_mp_key_doc)


with open(rf"{output_json}/{book}/mp_key_doc.json", "w") as outfile:
    json.dump(mp_key_doc, outfile)

with open(rf"{output_json}/{book}/keyword_paragraph_idx_map.json", "w") as outfile:
    json.dump(keyword_paragraph_idx_map, outfile)

with open(rf"{output_json}/{book}/scored_mp_key_doc.json", "w") as outfile:
    json.dump(scored_mp_key_doc, outfile)


## Saving paragraphs
paragraphs = {}
for keyword_idx, val in scored_mp_key_doc.items():
    paragraph = val[0]
    para_idx = val[2]
    paragraphs[para_idx] = paragraph

with open(f"{output_json}/{book}/paragraphs.json", "w") as content:
    json.dump(paragraphs, content)

