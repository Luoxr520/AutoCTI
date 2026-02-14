import json

# 用于计算语言模型（LLM）的资源使用情况，包括输入和输出的 token 数量和费用
class UsageCalculator:
    def __init__(self, response) -> None:
        self.response = response
        self.model = response.model

    def calculate(self):
        #import menu
        # 打开 Toolbox/menu/menu.json 文件并加载不同模型的定价信息
        with open ("Toolbox/menu/menu.json", "r") as f:
            data = json.load(f)
        iprice = data[self.model]["input"]
        oprice = data[self.model]["output"]
        usageDict = {}

        # 计算输入资源使用情况
        usageDict["model"] = self.model
        usageDict["input"] = {
            "tokens": self.response.usage.prompt_tokens,
            "cost": iprice*self.response.usage.prompt_tokens
        }

        # 计算输出资源使用情况
        usageDict["output"] = {
            "tokens": self.response.usage.completion_tokens,
            "cost": oprice*self.response.usage.completion_tokens
        }

        # 计算总资源使用情况
        usageDict["total"] = {
            "tokens": self.response.usage.prompt_tokens+self.response.usage.completion_tokens,
            "cost": iprice*self.response.usage.prompt_tokens+oprice*self.response.usage.completion_tokens
        }
        return usageDict