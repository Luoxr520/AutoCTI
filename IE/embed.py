# -*- coding: utf-8 -*-
from openai import OpenAI  # 新版SDK导入方式
import json
import time
import tiktoken
import os

# 初始化客户端
client = OpenAI(
    api_key="sk-JK0wel63Y0VfNVw57e1a9d23Cb3941D28c29769612100f16",
    base_url="https://openkey.cloud/v1"  # 指定代理地址
)

# 初始化Token编码器
encoding = tiktoken.get_encoding("cl100k_base")

# 定义输入和输出目录
INPUT_DIR = "D:\BBBB\graduation project\project\Data\Automotive-cyber-threat-intelligence-corpus-main\data\data"  # 存放文档的文件夹（如document1.txt, document2.txt等）
OUTPUT_DIR = "D:\BBBB\graduation project\project\Data\Automotive-cyber-threat-intelligence-corpus-main\data\embed data"

def generate_embedding(text: str) -> dict:
    """
    调用OpenAI API生成文本嵌入向量，并记录时间和Token消耗。
    返回包含embedding数组、时间和Token的字典。
    """
    start_time = time.time()
    try:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            extra_headers={"x-proxy-version": "2025-04-07"}  # 代理服务特殊头（按需添加）
        )
        embedding = response.data[0].embedding
    except Exception as e:
        print(f"生成嵌入时出错: {e}")
        embedding = []
    
    return {
        "embedding": embedding,
        "time": time.time() - start_time,
        "token": len(encoding.encode(text))
    }

def process_documents():
    """遍历输入目录下的所有txt文件，转换为JSON格式"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 遍历输入目录中的每个txt文件
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(INPUT_DIR, filename)
            doc_id = os.path.splitext(filename)[0]  # 如document1, document2
            
            # 读取文本内容
            with open(file_path, "r", encoding="utf-8") as f:
                cti_content = f.read().strip()
            
            # 设置元数据
            metadata = {
                "link": "NULL",  
                "annotated": 0  
            }
            
            # 计算字数
            word_count = len(cti_content.split())
            
            # 生成嵌入向量
            vector_data = generate_embedding(cti_content)
            
            # 构建JSON结构
            output_json = {
                "CTI": cti_content,
                "link": metadata["link"],
                "annotated": metadata["annotated"],
                "word_count": word_count,
                "vector": {
                    "text-embedding-3-large": vector_data
                }
            }
            
            # 保存为JSON文件
            output_path = os.path.join(OUTPUT_DIR, f"{doc_id}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_json, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    process_documents()