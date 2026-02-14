from dashscope import Application # 与 OpenAI API 交互
import time
import json
import logging # 记录日志信息
LOG = logging.getLogger(__name__)
import requests # 与 Hugging Face API 交互,发送 HTTP 请求
from openai import OpenAI
import os
from AttrDict import AttrDict

class LLMCaller:
    def __init__(self, config, prompt):
        self.config = config  
        self.prompt = prompt

    # 调用 Hugging Face LLaMA 模型
    def query_llama(self, payload):
        API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf"
        # 设置 API 的 URL 和认证头（包含 API 密钥）
        headers = {"Authorization": "Bearer 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"}
        response = requests.post(API_URL, headers=headers, json=payload)
        # return response.json()
        print("")
        return response.json()

    # 调用 通义 模型
    def qwen_plus(self, input):
        print("调用qwen-plus-latest模型")
        client = OpenAI(
            api_key = os.getenv("DASHSCOPE_API_KEY"), 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        print("qwen-plus-latest正在响应")
        # print("input:",input)
        response = client.chat.completions.create(
            model=self.config.model,
            # messages=[{"role": "user", "content": str(payload)}],
            messages=input,
            extra_body={
                "enable_search": True
            }
        )      
        JSONResp = response.json()  # 将响应内容转换为 JSON 格式
        return  JSONResp

    def gpt_4_mini(self, input):
        print("调用gpt-4o-mini模型")
        client = OpenAI(
            api_key = "sk-JK0wel63Y0VfNVw57e1a9d23Cb3941D28c29769612100f16", 
            base_url = "https://openkey.cloud/v1"
        )
        print("qwen_plus正在响应")
        # print("input:",input)
        response = client.chat.completions.create(
            model=self.config.model,
            # messages=[{"role": "user", "content": str(payload)}],
            messages=input,
            extra_body={
                "enable_search": True
            }
        )      
        JSONResp = response.json()  # 将响应内容转换为 JSON 格式
        return  JSONResp
    
    def Deepseek(self, input):
        print("调用deepseek-r1-distill-qwen-32b模型")
        client = OpenAI(
            api_key = "sk-47f2d862adf74ade822f5db3c5a6a2c1", 
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        print("deepseek-r1-distill-qwen-32b正在响应")
        response = client.chat.completions.create(
            model=self.config.model,
            messages=input
        )      
        # 获取响应内容字符串
        content = response.choices[0].message.content
        # 去除可能的格式标记（如```json）
        content_json = content.strip("```json\n").strip("```")
        # 解析为JSON对象
        JSONResp = json.loads(content_json)
        return JSONResp

    def call(self):
        startTime = time.time()
        # print("采用模型：",self.config.model)
        if self.config.model == "qwen-plus-latest":
            # input = {"role": "user", "content": "Write me a paragraph about the history of the United States."}
            input = self.prompt
            input_str = json.dumps(input)
            payload = {
                'inputs': input_str,
            }
            response = self.qwen_plus(input)
            #print("response:",response)
            #print("response类型：",type(response))
            message = json.loads(response)
            #print("message:",message)
            #print("message类型：",type(message))
            response_obj = AttrDict(**message)
            #print("response_obj:",response_obj)
            content = message["choices"][0]["message"]["content"]
            content_josn = content.strip("```json\n").strip("```")

        

            JSONResp = json.loads(content_josn)
            #print("JSONResp:",JSONResp)
            #print("JSONResp类型：",type(JSONResp))

        elif self.config.model == "deepseek-r1-distill-qwen-32b":
            # input = {"role": "user", "content": "Write me a paragraph about the history of the United States."}
            input = self.prompt
            input_str = json.dumps(input)
            payload = {
                'inputs': input_str,
            }
            response = self.Deepseek(input)
            #print("response:",response)
            #print("response类型：",type(response))
            message = json.loads(response)
            #print("message:",message)
            #print("message类型：",type(message))
            response_obj = AttrDict(**message)
            #print("response_obj:",response_obj)
            content = message["choices"][0]["message"]["content"]
            content_josn = content.strip("```json\n").strip("```")
            JSONResp = json.loads(content_josn)
            #print("JSONResp:",JSONResp)
            #print("JSONResp类型：",type(JSONResp))

        else:
            attempts = 0
            max_attempts = 5
            success = False
            while attempts < max_attempts and not success:
                client = OpenAI(api_key=self.config.api_key)
                response = client.chat.completions.create(
                    model = self.config.model,
                    response_format = { "type": "json_object" },
                    messages = self.prompt,
                    max_tokens= 4096,
                )
                try:
                    JSONResp = json.loads(response.choices[0].message.content)
                    JSONResp["triplets"]
                    success = True
                except (json.decoder.JSONDecodeError, KeyError) as e:
                    attempts += 1
                    LOG.error(f"Attempt {attempts}: {str(e)}")
                    if attempts < max_attempts:
                        LOG.info("Retrying...")
                    else:
                        LOG.error("Maximum attempts reached. Failing...")
                        raise e 
                    
        endTime = time.time()
        generation_time = endTime - startTime

        return response_obj, generation_time, JSONResp