# #
# # deepseek-ai\DeepSeek-R1-Distill-Qwen-1.5B
# from flask import Flask
# import flask
# import json
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# # torch.set_num_threads(8)
# # torch.set_num_interop_threads(8)
# # import torch.multiprocessing as mp
# server = Flask(__name__)
# model_name = "deepseek_finetuned_on_bhagwad_Geeta"
# model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# # device_ids = list(range(torch.cuda.device_count()))
# # model = nn.DataParallel(model).cuda()
# # 初始化生成器
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer,num_workers=4,device_map="auto")
#
# @server.route("/deepseek", methods=["GET", "POST"])
# def generate_text_by_deep_seek_r1_distill() -> str:
#
#     # 获取用户输入的文本
#     conversation_history = []
#     input_text = flask.request.args.get('content')
#
#
#
#     # 将当前用户的输入添加到对话历史中
#     conversation_history.append({"role": "user", "content": input_text})
#
#     # 生成回复
#     outputs = generator(
#         conversation_history,  # 传递完整的对话历史
#         max_new_tokens=1024,
#         do_sample=True,
#         temperature=0.7,
#         pad_token_id=tokenizer.pad_token_id,
#     )
#     response_text = outputs
#     # # 获取生成的文本并提取回复内容
#     # response_text = outputs[-1]["generated_text"][-1]["content"]
#     # # response_text = '问题：\n{}\n回答：{}'.format(input_text, response_text)
#     # response_text = '回答：{}'.format(response_text)
#     # response_text = response_text.replace('\n', '<br>')  # 替换换行符为<br>标签
#
#
#
#     # 将模型的回复添加到对话历史
#     # conversation_history.append({"role": "assistant", "content": response_text})
#
#     # 返回生成的回复
#     return json.dumps(response_text, ensure_ascii=False)
#
#
# if __name__ == "__main__":
#
#     # output = generate_text_by_deep_seek_r1_distill(input_text)
#     # print(output)
#     server.run(host="0.0.0.0", port=5151,threaded=True)
#
#
# # python -m gunicorn -w 4 -b 0.0.0.0:5151 --daemon --pid gunicorn.pid chat:server
# # gunicorn -w 4 -b 0.0.0.0:5151 chat:server

# import pandas as pd
#
# data = pd.read_csv(r"C:\Users\11762\Desktop\fsdownload\新疆翻译1.csv", encoding='gbk')
# syslog_list, transaltion_list, deepseek_list = list(data['日志描述']), list(data['翻译']), list(data['DeepSeek翻译'])
#
# with open('syslog日志分析.txt','a') as w:
#     for syslog,transaltion,deepseek in zip(syslog_list, transaltion_list, deepseek_list):
#         w.write('原始日志文档：{}'.format(syslog))
#         w.write('\n')
#         w.write('新疆已上线模型翻译：{}'.format(transaltion))
#         w.write('\n')
#         w.write('DeepSeek-1.5B模型解析结果：')
#         w.write(deepseek)
#         w.write('\n')
#         w.write('~~'*50)
#         w.write('\n')

from multiprocessing import Pool, cpu_count
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from tqdm import tqdm
import torch

# 多卡设置
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,5"
tokenizer = AutoTokenizer.from_pretrained('ds321-32b')
model = AutoModelForCausalLM.from_pretrained('ds321-32b', device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# 生成函数（外部可调用）
def generate_response(line):
    input_text = f"直接输出解决方案，不要输出任何解释和思考过程,回答:{line}翻译日志，并给出日志中问题如何解决"
    conversation_history = [{"role": "user", "content": input_text}]
    try:
        outputs = generator(
            conversation_history,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.pad_token_id,
        )
        answer = outputs[-1]['generated_text'][-1]["content"]
        return answer.split('</think>')[1] if '</think>' in answer else answer
    except Exception as e:
        return f"[生成失败]: {e}"

if __name__ == '__main__':
    data = pd.read_csv('result/0.csv', encoding='gbk')
    save_length = 50
    num_processes = min(cpu_count(), 8)  # 控制并发进程数（例如最多 8 核）

    for i in range(0, data.shape[0] // save_length+1):
        split_data = data.iloc[save_length * i:save_length * (i + 1), :].copy()
        lines = split_data['日志描述'].tolist()

        # 多进程并行调用生成器
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(generate_response, lines), total=len(lines), desc="翻译进度"))

        split_data['DeepSeek翻译'] = results
        split_data.to_csv(f'deepseek_results/deepseek32B结果{i+1}.csv', encoding='gbk', errors='ignore', index=False)

