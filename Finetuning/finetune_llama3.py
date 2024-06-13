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




import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "meta-llama/Meta-Llama-3-8B", # Choose ANY! eg mistralai/Mistral-7B-Instruct-v0.2
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
#     token = "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"
# )
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

################################################### Data Preparation ###################################################
data = pd.read_csv('data/wiki_training_data_with_context.csv')
dataset = datasets.Dataset.from_pandas(data)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an AI assistant in writing Wikipedia articles on personalities and your task is to expand the existing content of the given Wikipedia section about the personality: {} from the given context. Using the context generate a coherent, insightful and neutral expansion of the existing content. Strictly DO NOT use any external information.

### Input:
Existing Content: 
{}: {}

### Context:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    texts = []
    for person, section_title, incomplete_content, context, output in zip(examples['title'], examples["section"], examples["incomplete_content"], examples["context"], examples["last_paragraph"]):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(person, section_title, incomplete_content, context, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

dataset = dataset.map(formatting_prompts_func, batched = True,)
#########################################################################################################################

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
    # load_in_8bit_fp32_cpu_offload=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


peft_args = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    # target_modules=["gate_proj","q_proj","lm_head","o_proj","k_proj","embed_tokens","down_proj","up_proj","v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = SFTConfig(
    output_dir='saved_model/',
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
#     evaluation_strategy="epoch",
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=5,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
    )

peft_model = get_peft_model(model, peft_args)

with open("actual.txt", "w") as content:
    content.write(str(model))

with open("peft.txt", "w") as content:
    content.write(str(peft_model))

print(f"Peft model trainable parameters: {peft_model.print_trainable_parameters()}")

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
#     eval_dataset=test_dataset,
    # peft_config=peft_args,
    dataset_text_field="text",
    max_seq_length=704,
    tokenizer=tokenizer,
    args=training_params,
    packing=True,
)

trainer.train()

#model.push_to_hub("dparam/Llama3-Wiki-NPOV-finetuned")
trainer.model.save_pretrained("saved_model/llama-3-8B-instruct-5")
tokenizer.save_pretrained("saved_model/llama-3-8B-instruct-5")