# from text2vec import SentenceModel
from sentence_transformers import SentenceTransformer
import hnswlib
import numpy as np
import json
import glob
from read_data import read_data
'''
https://huggingface.co/shibing624/text2vec-base-chinese/tree/main
pip install faiss-gpu  #注意faiss版本和numpy库版本对应
'''
# 使用 Sentence-BERT 模型
# model = SentenceModel('text2vec_base_chinese')
model = SentenceTransformer('all-MiniLM-L6-v2')
path_list = glob.glob("data/*.*")
texts = read_data(path_list)
text_embeddings = model.encode(texts)
text_embeddings = np.array(text_embeddings).astype('float32')

dim = text_embeddings.shape[1]  # 向量的维度（例如：384维）
num_elements = len(texts)
text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)
index = hnswlib.Index(space='l2', dim=dim)  # 使用 L2 距离
index.init_index(max_elements=num_elements, ef_construction=200, M=32)
index.add_items(text_embeddings)

# 保存索引
index.save_index("text_hnsw_index.bin")

# 保存文本数据
with open("documents.json", "w", encoding="utf-8") as f:
    json.dump(texts, f, ensure_ascii=False, indent=4)

with open('documents.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)

index = hnswlib.Index(space='l2', dim=dim)
index.load_index("text_hnsw_index.bin")

# 查找最近的 5 个向量
query_embedding = model.encode(["PTP8.4.113.11 PTP/6/SRCSWITCH日志信息PTP/6/SRCSWITCH: Time source change from [STRING1] to [STRING2]"])
query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True)
labels, distances = index.knn_query(query_embedding, k=1)
print(labels, distances)

print(f"Top 1: {documents[labels[0][0]]} (L2距离: {distances[0][0]})")