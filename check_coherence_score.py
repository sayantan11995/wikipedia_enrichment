import pickle
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
from sgnlp.models.coherence_momentum import CoherenceMomentumModel, CoherenceMomentumConfig, \
    CoherenceMomentumPreprocessor


# Load Model
model_path = "./CoherenceMomentumModel"
config = CoherenceMomentumConfig.from_pretrained(
    "CoherenceMomentumModel/config.json",
    local_files_only=True
)
coherence_model = CoherenceMomentumModel.from_pretrained(
    "CoherenceMomentumModel/pytorch_model.bin",
    config=config,
    local_files_only=True
)

# # Load Model
# config = CoherenceMomentumConfig.from_pretrained(
#     "https://storage.googleapis.com/sgnlp-models/models/coherence_momentum/config.json"
# )
# coherence_model = CoherenceMomentumModel.from_pretrained(
#     "https://storage.googleapis.com/sgnlp-models/models/coherence_momentum/pytorch_model.bin",
#     config=config
# )
# coherence_model = CoherenceMomentumModel.from_pretrained(model_path)
coherence_model.to("cuda")

preprocessor = CoherenceMomentumPreprocessor(config.model_size, config.max_len)


with open("config.json", "r") as content:
    config = json.load(content)

with open("book_config.json", "r") as content:
    book_config = json.load(content)

f = open("out.txt", "w")

section_wise_coherence_score_diff = []
for book in book_config.keys():
# book = config.get("book")
    output_json = "output_json"
    dir_path = rf"books/{book}/part"
    rootdir = rf"books/{book}/"
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


    for items in top_section_keywords_mapping:
        text = keyword_paragraph_map[str(items[1])]
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
        # if difference > -0.5:
        #     f.write("BEGINNNNNNNNNNNNNNNNNNNNNNNN\n" + old_text+ "\n################\n" + text + "\n" + "ENDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD\n" )
        #     print(old_text)
        #     print("################")
        #     print(text)

f.close()


# Plotting the distribution
plt.hist(section_wise_coherence_score_diff, bins=20, color='blue', edgecolor='black')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('coherence_score_difference')

# Saving the plot as a figure
plt.savefig("coherence_score_diff.png")

# updated_wikipedia_content = " ".join(section_id_to_section_content.values())


# old_score = utils.calculate_quality(old_wikipedia_content)
# updated_score = utils.calculate_quality(updated_wikipedia_content)

# # Calculate the difference between corresponding values
# difference_dict = {key: updated_score[key] - old_score[key] for key in old_score}

# print(difference_dict)
