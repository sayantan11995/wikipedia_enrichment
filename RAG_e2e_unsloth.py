### This file can be used using the model loaded by unsloth FastLanguageModel

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

with open("config.json", "r") as content:
    config = json.load(content)

with open("book_config.json", "r") as content:
    book_config = json.load(content)

finetuned_llama3_results = {"tesseract": {},
                     "langchain": {}}

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# save_path = "Finetuning/saved_model/finetuned_llama-3-8b-wikipedia-section-completion-20240606T103921Z-001/finetuned_llama-3-8b-wikipedia-section-completion/"
# save_path = "Finetuning/saved_model/unsloth_finetuned_llama-3-8b-wikipedia-section-completion/"
save_path = "Finetuning/saved_model/unsloth_finetuned_llama-3-8b-wikipedia-section-completion_continue/"

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = save_path, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # local_files_only=True
    )
FastLanguageModel.for_inference(model) # Enable native 2x faster inference


###################################################################### For base model
# alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# You are an AI assistant in writing Wikipedia articles on personalities and your task is to expand the existing content of the given Wikipedia section about the personality: {} from the given context. Using the context generate a coherent, insightful and neutral expansion of the existing content. Strictly DO NOT use any external information.

# ### Input:
# Existing Content: 
# {}: {}

# ### Context:
# {}

# ### Response:
# {}"""
#########################################################################################

##################################################################### For instruct model
prompt_template = """You are an AI assistant in writing Wikipedia articles on personalities and your task is to expand the existing content of the given Wikipedia section about the personality: "{}" from the given context. Using the context generate a coherent, insightful and neutral expansion of the existing content. DO NOT use first person words such as "I", "my". DO NOT use any external information. If it is not possible to expand the content from the context, say so.

Context: "{}"

Existing content: "{}: {}"

Generated content: """

########################################################################################


for book, config in book_config.items():
    # if book != "John_Hall":
    #     continue
    for ocr_type in ["tesseract", "langchain"]:
        # book = config.get("book")
        # book_path = config.get("book_path") # whether to choose tesseract or langchain OCR
        book_path = f"{ocr_type}_OCRed"
        output_json = "output_json"
        dir_path = rf"books/{book_path}/{book}/part"
        rootdir = rf"books/{book_path}/{book}/"
        wiki_url = config.get("wikipedia_link")
        person = book.replace("_", " ")

        print(book)
        print(rootdir)

        if not os.path.exists(rootdir):
            finetuned_llama3_results[ocr_type][book] = {}
            continue


        Path(rf"{output_json}/{book}").mkdir(parents=True, exist_ok=True)

        loader = DirectoryLoader(rootdir, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        data=loader.load()

        #######################
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", "."],)
        all_splits = text_splitter.split_documents(data)
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cuda"}
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        #######################

        if len(all_splits) <= 0:
            finetuned_llama3_results[ocr_type][book] = {}
            continue
        print(all_splits[0])
        print("()"*50)

        shutil.rmtree("chroma_db", ignore_errors=True)
        vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

        # from transformers import TextStreamer
        # text_streamer = TextStreamer(tokenizer)

        # llm = HuggingFacePipeline(pipeline=pipe)

        # Retriving Top n Chunks most Similar to query.
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        # retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})

        #######################

        # def format_docs(docs):
        #     return "\n\n".join(doc.page_content for doc in docs)

        try:
            wikipedia_content = utils.get_wikipedia_content(wiki_url)
        except:
            finetuned_llama3_results[ocr_type][book] = {}
            continue

        ## Iterate for best of 3 result

        best_so_far = -10000000
        for iter in range(3):

            generated_content = {}
            for section_name, section_content in tqdm(wikipedia_content.items()):

                print(f"Processig section: {section_name}")

                query_stage_1 = f"""{section_name}: {section_content}"""

                # result = qa({"query": query_stage_1})


                ## Using the retrieved document as context to query the LLM 
                # context = format_docs(result.get("source_documents", []))
                context = "\n".join([doc.page_content for doc in retriever.get_relevant_documents(query_stage_1)])
                print(len(retriever.get_relevant_documents(query_stage_1)))
                # print(f"Retrieved Context: {context}")

                ################################################ For base llama model ################################################
                # query_stage_2 = alpaca_prompt.format(
                #         person,
                #         section_name,
                #         section_content,
                #         context,
                #         "", # output - leave this blank for generation!
                #     )

                # inputs = tokenizer(
                # [
                #     query_stage_2
                # ], return_tensors = "pt").to("cuda")

                # input_ids = inputs["input_ids"]
                # terminators = [
                #         tokenizer.eos_token_id,
                #         tokenizer.convert_tokens_to_ids("<|eot_id|>")
                #     ]

                # print(input_ids.shape)
                # outputs = model.generate(
                #     input_ids,
                #     max_new_tokens=150,
                #     eos_token_id=terminators,
                #     do_sample=True,
                #     temperature=0.6,
                #     top_p=0.9,
                # )
                #######################################################################################################
                tokenizer = get_chat_template(
                    tokenizer,
                    chat_template = "llama-3", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
                    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
                )

                query_stage_2 = prompt_template.format(
                        person,
                        context,
                        section_name,
                        section_content,
                        "", # output - leave this blank for generation!
                    )

                messages = [
                    {"from": "human", "value": query_stage_2},
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
                predicted_answer = "".join(generated_text)

                print(predicted_answer)
                
                # output_text = tokenizer.decode(outputs[0], skip_special_tokens = True)
                # generated_content[section_name] = truncate_to_last_sentence(output_text[len(query_stage_2):].strip())
                generated_content[section_name] = truncate_to_last_sentence(predicted_answer.strip())



            old_wikipedia_content = " ".join(wikipedia_content.values())

            


            text = ""
            for section_name, text in generated_content.items():
                print(f"Section: {section_name}")
                print("Generated content: ")
                print(text)
                print("="*20)
                unable_to_generate = False
                for impossible_terms in ["I cannot generate", "impossible to generate", "not possible", "unable to generate", "not enough information", "impossible to expand"]:
                    if impossible_terms in text.lower():
                        unable_to_generate = True
                        break
                if not unable_to_generate:
                        print(text)
                        print("="*20)
                        wikipedia_content[section_name] += text

            with open(f"{output_json}/{book}/e2e_RAG_generated_content_llama3_instruct_{ocr_type}.json", "w") as content:
                json.dump(generated_content, content)

            updated_wikipedia_content = " ".join(wikipedia_content.values())

            old_score = utils.calculate_quality(old_wikipedia_content)

            print(f"Old score: {utils.calculate_quality(old_wikipedia_content)}")
            updated_score = utils.calculate_quality(updated_wikipedia_content)
            print(f"Updated score: {updated_score}")

            # Calculate the difference between corresponding values
            difference_dict = {key: updated_score[key] - old_score[key] for key in old_score}

            curr_result = difference_dict["understandability"] + difference_dict["readability"] # comparing with readability+understandability

            print(curr_result, best_so_far)
            if curr_result > best_so_far:
                best_so_far = curr_result
                best_difference_dict = difference_dict.copy()

        print(f"last result: {difference_dict}")
        print(f"Best: {best_difference_dict}")

        finetuned_llama3_results[ocr_type][book] = best_difference_dict
    torch.cuda.empty_cache()
    # break


# with open("/content/finetuned_results.json", "w") as content:
#     json.dump(finetuned_llama3_results, content)


with open("results/finetuned_llama3_instruct_continue_results.json", "w") as content:
    json.dump(finetuned_llama3_results, content)

