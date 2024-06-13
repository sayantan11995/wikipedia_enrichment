import json
import requests
import re
import wikipedia
import pandas as pd
import shutil
import time
from tqdm import tqdm
from langchain.embeddings import HuggingFaceEmbeddings
from ast import literal_eval
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from bs4 import BeautifulSoup
import sys

if len(sys.argv) < 2:
    print("Usage: python data_preparation.py <DB_name> Default: 'chromaDB' <save_file> <data_start data_end>")
    db_name = 'chromaDB'
    save_file = 'wiki_training_data_with_context_full.csv'
else:
    db_name = sys.argv[1]
    save_file = sys.argv[2]
    data_start = sys.argv[3]
    data_end = sys.argv[4]

print(f"Using persistent db: {db_name}")


# data = pd.read_csv('data/wikipedia_fa_biographical_data_RAG.csv')
# data = data.head(2000)

# data = data.sample(20000, random_state=42)

def remove_square_brackets(text):
    pattern = r'\[.*?\]'
    return re.sub(pattern, '', text)

def RetreiveParagraphsAndLinksBetweenSections(current_section, next_section, bs_object):
    """
    Return:
    incomplete_content: all the paragrphs except last one
    last_paragraph: the last paragraph
    external_content: total page contents from the list of wikipedia links in the last paragraph
    """
    hElem = bs_object.find("span", {'id': current_section})
    endElem = bs_object.find('span', {'id': next_section})
    all_tags = list(bs_object.find_all())

    paragraph_list = []
    inBetween = False
    temp = []
    for tag in all_tags:
        if tag == hElem:
            inBetween = True
        if inBetween == True and tag.name == 'p':
            temp.append(tag)
            paragraph_list.append(tag.get_text())

        if tag == endElem:
            inBetween = False
            break

    if len(paragraph_list) == 0:
        return None, None, None

    incomplete_content = " ".join(paragraph_list[:-1])
    incomplete_content = remove_square_brackets(incomplete_content)

    last_paragraph = paragraph_list[-1]
    last_paragraph = remove_square_brackets(last_paragraph)

    wiki_links = []
    external_content = ""

    for tag in temp:
        for link in tag.find_all('a'):
            if 'wiki' in link.get('href'):
                external_page = link.get('href').split('/')[-1]
                # try:
                #     external_content += "\n\n" + wikipedia.page(external_page, auto_suggest=False).content
                # except:
                #     print(link.get('href'))
                #     print(external_page)
                #     pass
                wiki_links.append(external_page)
                # wiki_links.append(f"https://en.wikipedia.org/" + link.get('href'))

    # if external_content == "":
    #     return None, None, None
    return incomplete_content, last_paragraph, wiki_links



title = "elon_musk"

def create_data_with_links(title):
    data = []
    wiki_url = f"https://en.wikipedia.org/wiki/{title}"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    response = requests.get(wiki_url, headers=headers)
    #print(response.status_code)
    soup = BeautifulSoup(response.content, 'html.parser')

    wikipedia_section_names = []
    non_relevant_sections = ["See_also", "References", "Footnotes", "External_links", "Notes", "Bibliography", "Further_reading"]
    for link in soup.find_all('span', attrs={'class':'mw-headline'}):
        if link.get('id') is not None and link.get('id') not in non_relevant_sections:
            wikipedia_section_names.append(link.get('id'))
    print(wikipedia_section_names)

    
    for i in range(len(wikipedia_section_names)-1):
        try:
            incomplete_content, last_paragraph, wiki_links = RetreiveParagraphsAndLinksBetweenSections(wikipedia_section_names[i], wikipedia_section_names[i+1], soup)
            
            # if incomplete_content and last_paragraph and external_content:
            if wiki_links:
                data.append({
                    "title": title,
                    "section": wikipedia_section_names[i],
                    "incomplete_content": incomplete_content,
                    "last_paragraph": last_paragraph,
                    "wiki_links": wiki_links
                })
        except:
            pass

    return data


############################################### Run only if you don't have saved links ##############################################
# with open("biographical_pages.json", 'r') as f:
#     biographical_pages = json.load(f)

# wikipedia_fa_biographical_data_RAG = []
# count = 0
# for title in biographical_pages:
#     data = create_data_with_links(title)
#     print(len(wikipedia_fa_biographical_data_RAG))
#     wikipedia_fa_biographical_data_RAG.extend(create_data_with_links(title))
#     count += 1

#     # if count>2:
#     #     break

# print(len(wikipedia_fa_biographical_data_RAG))

# wikipedia_fa_biographical_data_RAG_df = pd.DataFrame(wikipedia_fa_biographical_data_RAG)

# print(wikipedia_fa_biographical_data_RAG_df.shape)
# wikipedia_fa_biographical_data_RAG_df.to_csv('data/wikipedia_fa_biographical_data_RAG.csv')
# ######################################################################################################################################

# wikipedia_fa_biographical_data_RAG_df = pd.read_csv('data/wikipedia_fa_biographical_data_RAG.csv', names=['title', 'section', 'incomplete_content', 'last_paragraph', 'wiki_links'])
wikipedia_fa_biographical_data_RAG_df = pd.read_csv('data/wikipedia_fa_biographical_data_RAG.csv')

print(wikipedia_fa_biographical_data_RAG_df.shape)
# print(wikipedia_fa_biographical_data_RAG_df.columns)

# data = wikipedia_fa_biographical_data_RAG_df.sample(20000, random_state=42)
data = wikipedia_fa_biographical_data_RAG_df.copy()

if data_start and data_end:
    data = data[int(data_start):int(data_end)]

print(data.shape)
print(data.index)

#######################
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", "."],)
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
#######################


for idx, rows in tqdm(data.iterrows()):

    external_content = ""

    links = literal_eval(rows['wiki_links'])
    for external_page in links:
        try:
            external_content += "\n" + wikipedia.page(external_page, auto_suggest=False).content

            # Remove section names using regex (=== Name ===)
            external_content = re.sub(r'==.*?==', '', external_content)
            external_content = re.sub(r'\n\n', '\n', external_content)

            # Remove non alpha numeric characters from text
            external_content = re.sub(r'[^a-zA-Z0-9\s]', '', external_content)
        except:
            pass


    all_splits = text_splitter.split_text(external_content)
    shutil.rmtree(db_name, ignore_errors=True)
    # time.sleep(1)
    print(len(all_splits))
    if len(all_splits) == 0:
        data.loc[idx, 'context'] = None
        continue

    try:
        vectordb = Chroma.from_texts(texts=all_splits, embedding=embeddings, persist_directory=db_name)
        # vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

        # Retriving Top n Chunks most Similar to query.
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})

        ## Retriving relevant context for the last section
        if rows['last_paragraph']:
            context = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(rows['last_paragraph'])])

        # print(len(context))
        data.loc[idx, 'context'] = context
    except Exception as e:
        print(f"Error occurred: {e}")
        pass

data = data[~data['context'].isna()]
print(data.shape)
print(data.isna().sum())

data.to_csv(f'data/{save_file}', index=False)