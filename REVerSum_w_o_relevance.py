import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from pathlib import Path
import langchain
import json
import shutil
import chromadb
from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings # Use this if getting ValueError: alternative_import must be a fully qualified module path !langchain-huggingface==0.0.2
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from tqdm import tqdm
import glob
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

import utils

# Function to truncate the text to the last complete sentence
def truncate_to_last_sentence(text):
    last_period_index = text.rfind('.')
    if last_period_index != -1:
        return text[:last_period_index + 1]
    else:
        return text


with open("book_config.json", "r") as content:
    book_config = json.load(content)



max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# save_path = "Finetuning/saved_model/finetuned_llama-3-8b-wikipedia-section-completion-20240606T103921Z-001/finetuned_llama-3-8b-wikipedia-section-completion/"
# save_path = "Finetuning/saved_model/unsloth_finetuned_llama-3-8b-wikipedia-section-completion/"
# save_path = "Finetuning/saved_model/unsloth_finetuned_llama-3-8b-wikipedia-section-completion_continue/"
save_path = "meta-llama/Meta-Llama-3-8B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = save_path, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # local_files_only=True
    )
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

##################################################################### For instruct model
relevance_prompt_template = """You are an expert in editing Wikipedia biography articles from external resources and you are assigned to expand the content of the given Wikipedia section about the personality: "{}" from the given context. You are provided with a section content below which requires to be expanded:

Section title: {}
Section content: {}

Based on the above content, I gathered some documents below:

Document 1: {}
Document 2: {}
Document 3: {}

As an expert, can you tell me out of these documents which document(s) is/are relevant to the above section content? Mention the document Id(s) without any explanation. If you feel no document from the above list are relevant, just say "No documents are relevant".
"""

# summary_template = """Can you extract the phrases only from the relevant document(s) which can be incorporated in the mentioned section? Make your response as informative as possible."""

summary_template = """As an expert in Wikipedia editor, can you make a consize summary from the given supporting statements, which can be seamlessly integrated with the mentioned section? Make your response as informative as possible without any duplicate information from the original content. Just response the summary without any further details. If you feel that it is not possible to generate a consize summary, say "Not possible."

Supporting statements:
{}
"""

evidence_extraction_template = """You are an expert in editing Wikipedia biography articles from external resources and you are assigned to expand the content of the given Wikipedia section about the personality: "{}" from the given context. You are provided with a section content below which requires to be expanded:

Section title: {}
Section content: {}

Based on the above content, I gathered some documents below:

Document 1: {}
Document 2: {}
Document 3: {}

As an expert in Wikipedia editor, can you extract the evidences from these documents, which can be seamlessly integrated with the mentioned section?  Just response the supporting statements as numbered list without any further details. Format should be - <1. Supporting statement 1>\n<2. Supporting statement 2>. If you feel that there is no supporting statement, say "No supporting statement."
"""


supporting_statement_verification_template = """You are an expert at document reviewing and you are assigned to review whether the given list of statements are extracted from the below documents

Statements:
{}

From the above statements can you tell me which statements are actually extracted from the below documents:

Document 1: {}
Document 2: {}
Document 3: {}

Output format should be - <statement number. statement>. If there is no statement extracted from the mentioned documents say "None." 
"""


########################################################################################
book = list(book_config.keys())[1]

# text_file = open("generated_contents.txt", "w")
text_file = open("REVerSum_w_o_relevance.txt", "w")

verified_statements_summary_results = {"tesseract": {},
                     "langchain": {}}

for book, config in book_config.items():
    # ocr_type = "tesseract"
    # book = "Jessie_Ackermann"
    for ocr_type in ["tesseract", "langchain"]:

        book_path = f"{ocr_type}_OCRed"
        output_json = "output_json"
        dir_path = rf"books/{book_path}/{book}/part"
        rootdir = rf"books/{book_path}/{book}/"
        wiki_url = book_config.get(book).get("wikipedia_link")
        person = book.replace("_", " ")

        print(book)
        print(rootdir)

        if not os.path.exists(rootdir):
            verified_statements_summary_results[ocr_type][book] = {}
            continue

        loader = DirectoryLoader(rootdir, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        data=loader.load()

        #######################
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", "."],)
        all_splits = text_splitter.split_documents(data)
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cuda"}
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        #######################

        shutil.rmtree("chroma_db", ignore_errors=True)
        vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

        # from transformers import TextStreamer
        # text_streamer = TextStreamer(tokenizer)

        # llm = HuggingFacePipeline(pipeline=pipe)

        # Retriving Top n Chunks most Similar to query.
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})


        try:
            wikipedia_content = utils.get_wikipedia_content(wiki_url)
        except:
            verified_statements_summary_results[ocr_type][book] = {}
            continue

        generated_content = {}


        text_file.write(f"=======================================Person:  {person} =================================================\n\n")
        text_file.write(f"=======================================OCR Type:  {ocr_type} =================================================\n\n")

        for section_name, section_content in tqdm(wikipedia_content.items()):

            print(f"Processig section: {section_name}")

            query_stage_1 = f"""{section_name}: {section_content}"""

            # result = qa({"query": query_stage_1})

            text_file.write(f"=====Existing section: {section_name}=====\n")
            text_file.write(f"{section_content}\n\n")


            ## Using the retrieved document as context to query the LLM 
            # context = format_docs(result.get("source_documents", []))
            context = "\n".join([doc.page_content for doc in retriever.get_relevant_documents(query_stage_1)])
            relevant_docs = [doc.page_content for doc in retriever.get_relevant_documents(query_stage_1)]

            # print('==================================================Relevant docs:')
            text_file.write(f"=====Retrieved documents:\n")
            for idx, docs in enumerate(relevant_docs):
                text_file.write(f"Document {idx+1}: \n {docs}\n")

            print(relevant_docs)
            # print(len(retriever.get_relevant_documents(query_stage_1)))
            # retrieved_context[section_name] = context

            tokenizer = get_chat_template(
                            tokenizer,
                            chat_template = "llama-3", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
                            mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
                        )

            # query_stage_2 = prompt_template.format(
            #         person,
            #         context,
            #         section_name,
            #         section_content,
            #         "", # output - leave this blank for generation!
            #     )
            
            print('==================================================First query:')

            evidence_extraction_prompt = evidence_extraction_template.format(
                    person,
                    section_name,
                    section_content,
                    relevant_docs[0],
                    relevant_docs[1],
                    relevant_docs[2],
                    "", # output - leave this blank for generation!
                ) ### For context after section content 

            messages = [
                {"from": "human", "value": evidence_extraction_prompt},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize = True,
                add_generation_prompt = True, # Must add for generation
                return_tensors = "pt",
            ).to("cuda")

            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=120,
                # eos_token_id=terminators,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

            response = outputs[0][input_ids.shape[-1]:]
            generated_text = tokenizer.batch_decode(response, skip_special_tokens=True)
            supporting_statements = "".join(generated_text)

            print(truncate_to_last_sentence(supporting_statements.strip()))
            text_file.write(f"===== supporting statements:\n")
            text_file.write(f"{truncate_to_last_sentence(supporting_statements.strip())}\n\n")


            print("----------------------------------- Supoorting statement verification -----------------------------------")

            supporting_statement_verification_query = supporting_statement_verification_template.format(
                    supporting_statements,
                    relevant_docs[0],
                    relevant_docs[1],
                    relevant_docs[2],
                    "", # output - leave this blank for generation!
                ) ### For context after section content 
            messages = [
                {"from": "human", "value": supporting_statement_verification_query},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize = True,
                add_generation_prompt = True, # Must add for generation
                return_tensors = "pt",
            ).to("cuda")

            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=120,
                # eos_token_id=terminators,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

            response = outputs[0][input_ids.shape[-1]:]
            generated_text = tokenizer.batch_decode(response, skip_special_tokens=True)
            verified_supporting_statements = "".join(generated_text)

            print(truncate_to_last_sentence(verified_supporting_statements.strip()))
            text_file.write(f"===== Supoorting statement verification:\n")
            text_file.write(f"{truncate_to_last_sentence(verified_supporting_statements.strip())}\n\n")

            print("----------------------------------- Summary generation -----------------------------------")

            summary_generation_query = summary_template.format(
                    verified_supporting_statements,
                    "", # output - leave this blank for generation!
                ) ### For context after section content 
            messages = [
                {"from": "human", "value": evidence_extraction_prompt},
                {"from": "assistant", "value": supporting_statements},
                {"from": "human", "value": summary_generation_query},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize = True,
                add_generation_prompt = True, # Must add for generation
                return_tensors = "pt",
            ).to("cuda")

            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=120,
                # eos_token_id=terminators,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

            response = outputs[0][input_ids.shape[-1]:]
            generated_text = tokenizer.batch_decode(response, skip_special_tokens=True)
            generated_summary = "".join(generated_text)

            print(truncate_to_last_sentence(generated_summary.strip()))
            text_file.write(f"===== Generated Summary:\n")
            text_file.write(f"{truncate_to_last_sentence(generated_summary.strip())}\n\n")


            generated_content[section_name] = truncate_to_last_sentence(generated_summary.strip())



       

        old_wikipedia_content = " ".join(wikipedia_content.values())

        text = ""
        for section_name, text in generated_content.items():

            unable_to_generate = False
            for impossible_terms in ["I cannot generate", "impossible to generate", "not possible", "unable to generate", "not enough information", "impossible to expand"]:
                if impossible_terms in text.lower():
                    unable_to_generate = True
                    break
            if not unable_to_generate:
                wikipedia_content[section_name] += text

        with open(f"{output_json}/{book}/REVerSum_w_o_relevance_llama3_instruct_{ocr_type}.json", "w") as content:
            json.dump(generated_content, content)

        updated_wikipedia_content = " ".join(wikipedia_content.values())

        old_score = utils.calculate_quality(old_wikipedia_content)

        # print(f"Old score: {utils.calculate_quality(old_wikipedia_content)}")
        updated_score = utils.calculate_quality(updated_wikipedia_content)
        # print(f"Updated score: {updated_score}")

        # Calculate the difference between corresponding values
        difference_dict = {key: updated_score[key] - old_score[key] for key in old_score}

        verified_statements_summary_results[ocr_type][book] = difference_dict
        torch.cuda.empty_cache()

        print(difference_dict)
    # break

text_file.close()

with open("results/REVerSum_w_o_relevance_llama3_instruct.json", "w") as content:
    json.dump(verified_statements_summary_results, content)