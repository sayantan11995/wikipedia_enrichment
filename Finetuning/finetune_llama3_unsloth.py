import requests
import re
import pandas as pd
import shutil
import time
from tqdm import tqdm
from ast import literal_eval
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer, SFTConfig 
import transformers
import datasets

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported



max_seq_length = 1536 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "saved_path/unsloth_finetuned_llama-3-8b-wikipedia-section-completion"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id, # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map = "auto"
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


# tokenizer = AutoTokenizer.from_pretrained(model_id)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

################################################### Data Preparation ###################################################
data = pd.read_csv('data/wiki_training_data_with_context.csv')
dataset = datasets.Dataset.from_pandas(data)



# alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# You are an AI assistant in writing Wikipedia articles on personalities and your task is to expand the existing content of the given Wikipedia section about the personality: {} from the given context. Using the context generate a coherent, insightful and neutral expansion of the existing content. Strictly DO NOT use any external information.

# ### Input:
# Existing Content: 
# {}: {}

# ### Context:
# {}

# Generated Content: """

prompt_template = """You are an AI assistant in writing Wikipedia articles on personalities and your task is to expand the existing content of the given Wikipedia section about the personality: "{}" from the given context. Using the context generate a coherent, insightful and neutral expansion of the existing content. DO NOT use any external information. If it is not possible to expand the content from the context, say so.

Context: "{}"

Existing content: "{}: {}"

Generated content: """

EOS_TOKEN = tokenizer.eos_token
def generate_conversations(examples):
    conversations = []
    for person, section_title, incomplete_content, context, output in zip(examples['title'], examples["section"], examples["incomplete_content"], examples["context"], examples["last_paragraph"]):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = prompt_template.format(person, section_title, incomplete_content, context)
        conv = [{"from": "human", "value": text}, { "from": "gpt", "value": output}]

        conversations.append(conv)
    return {"conversations" : conversations, }
pass

dataset = dataset.map(generate_conversations, batched = True,)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

dataset = dataset.map(formatting_prompts_func, batched = True,)

with open("conv.txt", "w") as content:
    content.write(str(dataset[0]["conversations"]))

with open("tokenized.txt", "w") as content:
    content.write(str(dataset[0]["text"]))

#########################################################################################################################

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 6,
        warmup_steps = 9,
        num_train_epochs = 8,
        # max_steps = 2,
        learning_rate = 2e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        save_steps = 250,
        # resume_from_checkpoint = "outputs/checkpoint-500/",
    ),
)
trainer_stats = trainer.train(resume_from_checkpoint = "outputs/checkpoint-500/")



# #model.push_to_hub("dparam/Llama3-Wiki-NPOV-finetuned")
model_finetuned = "unsloth_finetuned_llama-3-8b-wikipedia-section-completion_continue"

save_path = f"saved_model/{model_finetuned}"
# trainer.model.save_pretrained(save_path)
model.save_pretrained(save_path) # Local saving