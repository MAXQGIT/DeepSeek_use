import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, GenerationConfig, \
  DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model


from tokenizer_text import get_tokenized_id
import os

# os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
tokenizer = AutoTokenizer.from_pretrained('DeepSeek-R1-Distill-Llama-8B')#, use_fast=False, trust_remote_code=True

model = AutoModelForCausalLM.from_pretrained('DeepSeek-R1-Distill-Llama-8B',  device_map="auto")#trust_remote_code=True, torch_dtype=torch.half,
model.config.use_cache = False

# 开启梯度
model.enable_input_require_grads()
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 任务类型，常用于因果语言模型
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # LoRA 矩阵的秩，控制训练参数量，常用值为 4 或 8
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理:控制更新幅度的超参数
    lora_dropout=0.1# Dropout 比例，防止过拟合
)

# model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
model = get_peft_model(model, config)


'''
原文链接：https://blog.csdn.net/FL1623863129/article/details/137763826
'''

args = TrainingArguments(
    output_dir="./output/DeepSeek_full",
    per_device_train_batch_size=2,  # 每个设备上的 batch size
    gradient_accumulation_steps=2,  # 梯度累积步数，减少显存占用
    logging_steps=10, # 记录日志的步数
    num_train_epochs=3,  # 训练轮数
    save_steps=100,  # 保存检查点的步数
    learning_rate=1e-4, # 学习率
    fp16=True,  # 开启半精度浮点数训练，减少显存使用
    save_total_limit=1,  # 限制保存的检查点数量，节省磁盘空间
    save_on_each_node=True,
    gradient_checkpointing=True,
    dataloader_num_workers=3,
    local_rank=-1
    #logging_dir="./logs"  # 设置日志文件夹
)



trainer = Trainer(
    model=model,
    args=args,
    train_dataset=get_tokenized_id('dataset/medical_o1_sft_Chinese.json'),
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# 1、用于保存训练模型 -- 后面在进行模型合并
#model.save_pretrained("./output/DeepSeek")
#trainer.save_model("./output/DeepSeek")  # 保存完整模型

# 2、直接合并模型开始。。。。。
# 将 adapter 合并进模型（去除 adapter 依赖）
model = model.merge_and_unload()
model.save_pretrained("DeepSeek_full")
tokenizer.save_pretrained("DeepSeek_full")

# 直接合并模型结束。。。。。

text = "现在你要扮演我的女朋友--嬛嬛, User：你是谁？今天去约会怎么样？"

inputs = tokenizer(f"User: {text}\n\n", return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)