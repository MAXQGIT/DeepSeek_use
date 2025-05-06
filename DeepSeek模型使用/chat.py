#
# deepseek-ai\DeepSeek-R1-Distill-Qwen-1.5B
from flask import Flask
import flask
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# torch.set_num_threads(8)
# torch.set_num_interop_threads(8)
# import torch.multiprocessing as mp
server = Flask(__name__)
model_name = "deepseek_model_15B"
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# device_ids = list(range(torch.cuda.device_count()))
# model = nn.DataParallel(model).cuda()
# 初始化生成器
generator = pipeline("text-generation", model=model, tokenizer=tokenizer,num_workers=4,device_map="auto")

@server.route("/deepseek", methods=["GET", "POST"])
def generate_text_by_deep_seek_r1_distill() -> str:

    # 获取用户输入的文本
    conversation_history = []
    input_text = flask.request.args.get('content')

    # 将当前用户的输入添加到对话历史中
    conversation_history.append({"role": "user", "content": input_text})

    # 生成回复
    outputs = generator(
        conversation_history,  # 传递完整的对话历史
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
    )
    # 获取生成的文本并提取回复内容
    response_text = outputs[-1]["generated_text"][-1]["content"]
    # response_text = '问题：\n{}\n回答：{}'.format(input_text, response_text)
    response_text = '回答：{}'.format(response_text)
    response_text = response_text.replace('\n', '<br>')  # 替换换行符为<br>标签



    # 将模型的回复添加到对话历史
    # conversation_history.append({"role": "assistant", "content": response_text})

    # 返回生成的回复
    return json.dumps(response_text, ensure_ascii=False)


if __name__ == "__main__":

    # output = generate_text_by_deep_seek_r1_distill(input_text)
    # print(output)
    server.run(host="0.0.0.0", port=5151,threaded=True)


# python -m gunicorn -w 4 -b 0.0.0.0:5151 --daemon --pid gunicorn.pid chat:server
# gunicorn -w 4 -b 0.0.0.0:5151 chat:server