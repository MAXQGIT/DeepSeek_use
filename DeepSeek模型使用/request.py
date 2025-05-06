import requests
import json
import pandas as pd
from tqdm import tqdm
url = "http://192.168.40.75:6060/translation"
data = pd.read_csv('新疆train.csv', encoding='gbk')

save_length = 100
for i in range(0, data.shape[0] // save_length + 1):
    deepseek_list = []
    split_data = data.iloc[save_length * i:save_length * (i + 1), :].copy()
    texts = split_data['日志描述'].to_list()
    # 构造 POST 请求，确保数据以字典形式传递
    response = requests.post(url, json={"texts": texts}, stream=True)
    new_list = []
    # 使用流模式请求
    if response.status_code == 200:
        print("逐条翻译结果：")
        for line in tqdm(response.iter_lines(decode_unicode=True),desc='翻译进度',unit='条'):
            if line.startswith("data: "):
                result_json = json.loads(line.replace("data: ", ""))
                new_list.append(result_json["translation"])

    else:
        print(f"请求失败，状态码：{response.status_code}")
    split_data['new'] = new_list
    split_data.to_csv('deepseek_results/test_result{}.csv'.format(i + 1), encoding='gbk', index=False)



