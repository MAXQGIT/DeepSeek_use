1.translation.py中将读取result/0.csv文件，翻译‘日志描述’列，日志描述列是英文的，结果将每50条依次保存到deepseek_results/deepseek32B结果i.csv文件中。

2.thread_flask_translation.py多线程部署DeepSeek模型。

3.request.py请求部署DeepSeek模型，将读取'新疆.CSV'中‘日志描述’列，逐个回答后保存，保存方式是每100个保存打deepseek_results/test_resulti.csv文件中。

4.create_faiss.py是使用中文词嵌入模型all-MiniLM-L6-v2构建知识库。将data文件夹中的pdf,txt,docx文件构建成文本索引。

5.LLM_agent.py是deepseek模型构建和中文词嵌入模型all-MiniLM-L6-v2构建的RAG智能体方法。

6.deepseek.py是微调DeepSeek官方模型，不是量化模型。 模型下载地址 https://huggingface.co/deepseek-ai

7.read_data.py是构建智能体时，读取pdf,txt,docx文件用到的程度

8.整个项目本人记着需要python>=3.11版本

9.其他程序是本人调试程序写的垃圾辅助测试代码，有兴趣的看看，没兴趣就不要管它
