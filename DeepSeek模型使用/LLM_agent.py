import hnswlib
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
import json
import pandas as pd
from flask import Flask
import json
import flask
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,5"
tokenizer = AutoTokenizer.from_pretrained('DeepSeek-R1-Distill-Llama-8B')
model = AutoModelForCausalLM.from_pretrained('DeepSeek-R1-Distill-Llama-8B',device_map="auto")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2').to('cuda:2')
device = 'cuda'

with open('documents.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)
dim = 384

hnsw_index = hnswlib.Index(space='l2', dim=dim)
hnsw_index.load_index('text_hnsw_index.bin')
hnsw_index.set_ef(200)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer,device_map="auto")
def search_and_answer(generator,query, topk=2, threshold=0.5):
    conversation_history = []
    query_embeddings = sentence_model.encode([query], convert_to_tensor=True, device=device)
    # query_embeddings = np.array(query_embeddings).astype('float32')
    query_embeddings = query_embeddings.detach().cpu().numpy().astype('float32')
    query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    labels, distances = hnsw_index.knn_query(query_embeddings, k=topk)
    if distances[0][0] < threshold:
        retrieved_text = documents[labels[0][0]]
        prompt = ('### 根据提供内容'
                  '{}'
                  '### 用中文回答问题'
                  '{}翻译日志，并给出日志中问题如何解决').format(retrieved_text, query)
    else:
        prompt = '用中文回答{}'.format(query)
    conversation_history.append({"role": "user", "content": prompt})
    outputs = generator(
        conversation_history,  # 传递完整的对话历史
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
    )
    response_text = outputs[-1]["generated_text"][-1]["content"]

    # response_text = '问题：<br>{}<br>知识库中内容：<br>{}<br>相似度阈值：<br>{}<br>传入模型内容：<br>{}<br>回答内容：<br>{}'.format(
    #     query, documents[labels[0][0]], distances[0][0], prompt, response_text.replace('\n', '<br>'))
    # conversation_history.append({"role": "assistant", "content": response_text})
    return response_text

# server = Flask(__name__)
#
#
# @server.route("/deepseek", methods=['GET', 'POST'])
# def chat():
#     input_text = flask.request.args.get('content')
#     response_text = search_and_answer(input_text)
#     return json.dumps(response_text, ensure_ascii=False)
#
#
if __name__ == '__main__':
    data_path = 'result/0.csv'
    data = pd.read_csv(data_path, encoding='gbk')
    save_length = 50
    for i in range(0, data.shape[0] // save_length):
        deepseek_list =[]
        split_data = data.iloc[save_length * i:save_length * (i + 1), :].copy()
        for query in tqdm(split_data['日志描述'],desc='数据处理进度'):
            answer = search_and_answer(generator,query)
            # answer = response[-1]['generated_text'][-1]["content"]
            if '</think>' in answer:
                deepseek_list.append(answer.split('</think>')[1])
            else:
                deepseek_list.append(answer)
        split_data['deepseek'] = deepseek_list
        split_data.to_csv('RAG_results/deepseek32B{}结果.csv'.format(i+1), encoding='gbk', errors='ignore', index=False)

    # server.run(host='0.0.0.0', port=5151)
