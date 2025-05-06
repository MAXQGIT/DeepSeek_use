from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from flask import Flask, request, jsonify, Response
import json
from concurrent.futures import ThreadPoolExecutor

# 设置使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,5"

# 加载模型
tokenizer = AutoTokenizer.from_pretrained('deepseek_model_15B')
model = AutoModelForCausalLM.from_pretrained('deepseek_model_15B', device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, num_workers=4, device_map="auto")

# 创建 Flask 应用
app = Flask(__name__)

# 翻译函数
def translation(conversation_history):
    outputs = generator(
        conversation_history,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.3,
        pad_token_id=tokenizer.pad_token_id,
    )
    answer = outputs[-1]['generated_text'][-1]["content"]
    return answer.split('</think>')[1] if '</think>' in answer else answer

# 接口
@app.route('/translation', methods=['POST'])
def batch_translate():
    input_texts = request.json.get('texts')
    if not input_texts or not isinstance(input_texts, list):
        return jsonify({'error': '请传入一个有效的texts列表'}), 400

    # 构造 prompt 和会话历史
    conversation_history = [[{"role": "user", "content":
        f"直接输出解决方案，不要输出任何解释和思考过程,回答:{line}翻译日志，并给出日志中问题如何解决"}]
                            for line in input_texts]

    def generate():
        with ThreadPoolExecutor(max_workers=min(6, len(conversation_history))) as executor:
            results = executor.map(translation, conversation_history)
            for translated in results:
                yield f"data: {json.dumps({'translation': translated}, ensure_ascii=False)}\n\n"

    return Response(generate(), mimetype='text/event-stream')


# 启动服务
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6060)
