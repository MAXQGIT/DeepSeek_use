import pandas as pd
import re
import random

'''
日志中ip地址脱敏：随机替换成其他数字
'''


def randomize_ip(ip):
    return ".".join(str(random.randint(1, 255)) for _ in ip.split("."))


def replace_all_ips(text):
    """替换文本中所有 IP 地址"""
    ips = re.findall(r'\d+\.\d+\.\d+\.\d+', text)  # 找到所有 IP 地址
    for ip in ips:
        text = text.replace(ip, randomize_ip(ip))  # 逐个替换
    return text


if __name__ == '__main__':
    data = pd.read_csv('data-xj.csv', encoding='gbk')
    data['new_日志描述'] = data['日志描述'].map(replace_all_ips)
    print(data.iloc[:100, :])
