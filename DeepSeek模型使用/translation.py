import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,5"
tokenizer = AutoTokenizer.from_pretrained('ds321-32b')
model = AutoModelForCausalLM.from_pretrained('ds321-32b', device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, num_workers=4, device_map="auto")


def translation(conversation_history):
    outputs = generator(
        conversation_history,  # 传递完整的对话历史
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.3,
        pad_token_id=tokenizer.pad_token_id,
    )
    answer = outputs[-1]['generated_text'][-1]["content"]
    if '</think>' in answer:
        return answer.split('</think>')[1]
    else:
        return answer


data = pd.read_csv('result/0.csv', encoding='gbk')

save_length = 50
for i in range(0, data.shape[0] // save_length):
    split_data = data.iloc[save_length * i:save_length * (i + 1), :].copy()
    deepseek_list = []
    lines = ["直接输出解决方案，不要输出任何解释和思考过程,回答:{}翻译日志，并给出日志中问题如何解决".format(line) for
             line in split_data['日志描述'].tolist()]
    conversation_historys = [{"role": "user", "content": line} for line in lines]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(translation, conversation_history) for conversation_history in conversation_historys]
        for f in tqdm(as_completed(futures), total=len(futures), desc='翻译进度'):
            deepseek_list.append(f.result())
    split_data['DeepSeek翻译'] = deepseek_list
    split_data.to_csv('deepseek_results/deepseek32B结果{}.csv'.format(i), encoding='gbk', errors='ignore', index=False)
