from keyphrase_vectorizers import KeyphraseCountVectorizer, KeyphraseTfidfVectorizer
from keybert import KeyBERT
import yake
from rakun2 import RakunKeyphraseDetector
import torch
from sentence_transformers import SentenceTransformer, util
import math
import re
import textstat
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import requests

topN = 10

def get_keyword_bert(docs):
    vectorizer = KeyphraseTfidfVectorizer(pos_pattern= '<N.*>+')
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(docs=docs, vectorizer=vectorizer, top_n=topN, stop_words='english', keyphrase_ngram_range=(2,2))
    return keywords

def get_keyword_yake(docs):
    
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, windowsSize=2,top = topN)
    keywords = kw_extractor.extract_keywords(docs)
    return keywords

def get_keyword_rakun(docs):
    hyperparameters = {"num_keywords": topN,
                    "merge_threshold": 1.1,
                    "alpha": 0.3,
                    "token_prune_len": 2}

    keyword_detector = RakunKeyphraseDetector(hyperparameters)
    keywords = keyword_detector.find_keywords(docs, input_type="string")
    return keywords

def clean_ocr_text(ocr_text):
    # Task 1: Remove lines with all capitalized letters (assumed headers)
    ocr_text = re.sub(r'^[A-Z\s]+$', '', ocr_text, flags=re.MULTILINE)

    # Task 2: Remove sentences containing less than 3 words
    ocr_text = re.sub(r'\b\w{1,2}\b[^.]*[.!?]', '', ocr_text)

    # Task 3: Remove irrelevant digits
    ocr_text = re.sub(r'\b\d+\b', '', ocr_text)

    ocr_text = ocr_text.replace("\n\n", "")
    ocr_text = ocr_text.replace("\n", "")

    # Split the text into lines
    lines = ocr_text.split('\n')

    # Filter out lines that contain all uppercase letters or contain less than 3 words
    filtered_lines = [line for line in lines if not (line.isupper() or len(line.split(" ")) <= 3)]

    # # Filter out lines that contain less than 3 words
    # filtered_lines = [line for line in filtered_lines if len(line.split(" ")) <= 3]

    # Join the remaining lines back into a single string
    result = '\n'.join(filtered_lines)

    print(result)
    return result

def split_sentence_from_text(text):
    # file_loc = rf"/content/content/{book}/part%s.txt"%key
    # text = ""
    # with open(file_loc,encoding="utf8") as f:
    #     text = f.read()

    # res = re.findall(r"[^.!?]+", text)
    res = sent_tokenize(text)
    # print(res)
    filtered_lines = [line for line in res if not len(line.split(" ")) <= 3]
    return filtered_lines

embedder = SentenceTransformer('all-MiniLM-L6-v2')
embedder.to("cuda")
# def create_representative_doc(corpus, query):
#     corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
#     query_embedding = embedder.encode(query, convert_to_tensor=True)

#     cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
#     paragraph = ""
#     scores = []
#     for i,score in enumerate(cos_scores):
#         if score > 0.3:
#             paragraph += (corpus[i]+'.')
#         scores.append(score.detach().cpu().numpy())

#     if len(scores) > 0:
#         return (paragraph, sum(scores)/len(scores))

#     else:
#         return (paragraph, 0)
    

from sgnlp.models.coherence_momentum import CoherenceMomentumModel, CoherenceMomentumConfig, \
    CoherenceMomentumPreprocessor

# # Load Model
# model_path = "./CoherenceMomentumModel"
# config = CoherenceMomentumConfig.from_pretrained(
#     "CoherenceMomentumModel/config.json"
# )
# coherence_model = CoherenceMomentumModel.from_pretrained(
#     "CoherenceMomentumModel/pytorch_model.bin",
#     config=config
# )
# # coherence_model = CoherenceMomentumModel.from_pretrained(model_path)
# coherence_model.to("cuda")

# preprocessor = CoherenceMomentumPreprocessor(config.model_size, config.max_len)

def create_representative_doc(corpus, query):
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    sorted_score_with_indices = sorted(enumerate(cos_scores), key=lambda x: x[1], reverse=True)
    sorted_score_with_indices = [(index, value.detach().cpu().numpy())  for index, value in sorted_score_with_indices]
    # sorted_indices = [(index, value.detach().cpu().numpy())  for index, value in sorted_indices]
    paragraph = corpus[sorted_score_with_indices[0][0]] # Initialize with the most similar sentence
    scores = [sorted_score_with_indices[0][1]]
    

    text_tensor = preprocessor([paragraph])
    coherence_score = coherence_model.get_main_score(text_tensor["tokenized_texts"].to("cuda")).item()

    for (index, score) in sorted_score_with_indices[1:20]:
        updated_paragraph = paragraph + "# " + corpus[index]
        text_tensor = preprocessor([updated_paragraph])
        updated_coherence_score = coherence_model.get_main_score(text_tensor["tokenized_texts"].to("cuda")).item()

        print("Difference: " + str(coherence_score - updated_coherence_score))
        if coherence_score - updated_coherence_score <= 35:
            paragraph = updated_paragraph
            scores.append(score)
        else:
            updated_paragraph = paragraph

    print("Paragraph_length: " + str(len(paragraph.split(" "))))
    if len(scores) > 0:
        return (paragraph, sum(scores)/len(scores))

    else:
        return (paragraph, 0)
    


def FetchParagraphBetweenIds(id1, id2, bs_object):
    hElem = bs_object.find("span", {'id': id1})
    endElem = bs_object.find('span', {'id': id2})
    cntns = list(bs_object.find_all())

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

def get_wikipedia_content(wiki_url):
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

    section_id_to_section_content = {}
    for i in range(len(wikipedia_section_names)-1):
        section_id_to_section_content[wikipedia_section_names[i]] = FetchParagraphBetweenIds(wikipedia_section_names[i], wikipedia_section_names[i+1], soup)
    
    return  section_id_to_section_content


def calculate_quality(text):
    # Informativeness
    num_characters = len(text)
    num_sentences = textstat.sentence_count(text)
    num_words = textstat.lexicon_count(text)
    num_complex_words = textstat.difficult_words(text)

    informativeness = 0 * num_characters + 0.151 * num_sentences + 0.154 * num_words + 0.155 * num_complex_words

    # print("Informativeness:  ", informativeness)

    # Readability
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    coleman_liau_index = textstat.coleman_liau_index(text)
    complex_words_percentage = num_complex_words / num_words
    average_syllables_per_word = textstat.syllable_count(text) / num_words

    readability = 0.213 * flesch_kincaid_grade + 0.185 * coleman_liau_index + 0.26 * complex_words_percentage + 0.253 * average_syllables_per_word

    # print("Readability:  ", readability)

    # Understandability
    gunning_fog_score = textstat.gunning_fog(text)
    smog_index = textstat.smog_index(text)
    automated_readability_index = textstat.automated_readability_index(text)
    average_words_per_sentence = num_words / num_sentences

    understandability = 0.393 * gunning_fog_score + 0.352 * smog_index + 0.181 * automated_readability_index + 0.344 * average_words_per_sentence

    # print("Understandability: ", understandability)

    # Calculate overall quality
    quality = 0.255 * informativeness + 0.654 * readability + 0.557 * understandability

    return {"informativeness": informativeness,
            "understandability": understandability,
            "readability": readability,
            "quality": quality}


def run_rag(retrievalQA, query):
    # print(f"Query: {query}\n")
    result = retrievalQA.run(query)
    # print("\nResult: ", result)
    return result

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def calculate_similarity(sentence_transformer_model, text1, text2):

    # Compute embedding for both lists
    embeddings1 = sentence_transformer_model.encode(text1, convert_to_tensor=True)
    embeddings2 = sentence_transformer_model.encode(text2, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scosimilarity_scoreres = util.cos_sim(embeddings1, embeddings2)[0][0].item()

    return cosine_scosimilarity_scoreres