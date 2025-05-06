'''
微调官方deepseek模型
'''

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from transformers import TrainingArguments, Trainer
import os

# os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
model_name = "DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
dataset = load_dataset("json", data_files="train_data.json", split='train[:-1]')
tts = dataset.train_test_split(test_size=0.1)
train_data = tts['train']
# test_data = tts['test']


def tokenize_function(examples):
    combined_texts = [f"{question}\n{complex}\n{answer}" for question, complex, answer in
                      zip(examples["Question"], examples["Complex_CoT"], examples["Response"])]
    tokenized = tokenizer(combined_texts, truncation=True, max_length=2048, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


train_data_tokenized = train_data.map(tokenize_function, batched=True)
# test_data_tokenized = test_data.map(tokenize_function, batched=True)
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# model = AutoModelForCausalLM.from_pretrained(model_name,  device_map="auto")

lora_config = LoraConfig(
    r=8,
    lora_alpha=8,  # lower = fast task
    # target_modules = ["q_proj" , "v_proj"],
    # lora_dropout = 0.05,
    # bias = "none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
training_args = TrainingArguments(
    output_dir="outputs",
    num_train_epochs=20,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    fp16=True,
    logging_steps=10,
    learning_rate=3e-5,
    dataloader_num_workers=4,
    local_rank=-1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data_tokenized,
    # eval_dataset=test_data_tokenized
)
trainer.train()
save_path = "deepseek_finetuned_on_bhagwad_Geeta"

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
