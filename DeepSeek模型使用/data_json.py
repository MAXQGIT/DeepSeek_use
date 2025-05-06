#
# import json
#
# train_data = []
# with open('专家日志.txt', 'r',encoding="gbk", errors="ignore") as r:
#     # print(chardet.detect(r.read())['encoding'])
#     data = r.read().split('父主题')
#     for line in data:
#         line_data = {}
#         if '日志含义' and '可能原因' and '日志参数' in line:
#             question, result = line.split('日志含义')[0], line.split('日志含义')[1]
#             line_data['Question'] = question.strip('\n').replace('：','')
#             complex_cot, response = result.split('可能原因')[0], result.split('可能原因')[1]
#             complex_cot = complex_cot.split('日志参数')[0]
#             line_data['Complex_CoT'] = complex_cot.strip('\n')
#             line_data['Response'] = response.strip('\n').replace('?','')
#             train_data.append(line_data)
# with open('train_data.json', 'w', encoding='utf-8') as f:
#     json.dump(train_data, f, ensure_ascii=False, indent=4)


# import json
# data_list = []
# with open('train_data.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#     for line_dict in data:
#         line = line_dict['Question'] + ' ' + line_dict['Complex_CoT'] + ' ' + line_dict['Response']
#         data_list.append(line)
# with open('操作手册.txt','w',encoding='utf-8') as w:
#     for line in data_list:
#         w.write(line.replace('\n',''))
#         w.write('\n')
#

# import json
# import pandas as pd
#
# syslog_list, translation_list = [], []
# with open('train_data.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#     for line_dict in data:
#         if '日志信息' in line_dict['Question']:
#             syslog_list.append(line_dict['Question'].split('日志信息')[1])
#         translation_list.append('日志含义\n' + line_dict['Complex_CoT'] + '\n' + '可能原因\n' + line_dict['Response'])
#
# data = pd.DataFrame({'日志描述': syslog_list, '日志解析': translation_list})
# data.to_csv('data.csv', index=False, encoding='gbk')

import pandas as pd
data = pd.read_csv('result/0.csv', encoding='gbk')
print(data)
print(data.shape)
# data = pd.read_csv('data.csv',encoding='gbk')
# for i in range(0, data.shape[0] // 500+1):
#     split_data = data.iloc[500 * i:500 * (i + 1), :].copy()
#     split_data.to_csv('result/{}.csv'.format(i),encoding='gbk',index=False)
