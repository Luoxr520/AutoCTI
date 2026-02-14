from openai import OpenAI
import time
from omegaconf import DictConfig
import os

class LLMCaller:
    def __init__(self, config: DictConfig, prompt) -> None:
        self.config = config
        self.prompt = prompt

    def call(self):
        client = OpenAI(
            api_key = os.getenv("DASHSCOPE_API_KEY"), 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        startTime = time.time()
        response = client.chat.completions.create(
            model = self.config.model,
            response_format = { "type": "json_object" },
            messages = self.prompt,
            max_tokens = 2048,
            # stream=True  # 设置流式响应
        )
        endTime = time.time()
        #pause for 5 seconds to avoid exceeding the rate limit
        time.sleep(5)
        generation_time = endTime - startTime
        return response, generation_time

