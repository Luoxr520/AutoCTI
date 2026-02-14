import json
from usageCalculator import UsageCalculator
import logging
LOG = logging.getLogger(__name__)

class ResponseParser:
    def __init__(self, llmAnnotator) -> None:
        #print("llmAnnotator.JSONResp:",llmAnnotator.JSONResp)
        self.llm_response = llmAnnotator.llm_response
        self.prompt = llmAnnotator.prompt
        self.query = llmAnnotator.inFileJSON["CTI"] # 过滤后的文本
        self.link = llmAnnotator.inFileJSON["link"]
        self.config = llmAnnotator.config
        self.JSONResp = llmAnnotator.JSONResp
    def parse(self):
        # 解析 LLM 的输出

        # 尝试解析LLM的响应内容为JSON：如果解析失败，记录错误日志并抛出异常。       
        # try:
        #     responseJSON = json.loads(self.llm_response.choices[0].message.content)
        # except json.decoder.JSONDecodeError:
        #     LOG.error("Error decoding JSON")
        #     LOG.info(f"LLM's output: {self.llm_response.choices[0].message.content}")
        #     raise

        self.output = {
            "CTI": self.query,
            "annotator": self.JSONResp,
            "link": self.link,
            "usage": UsageCalculator(self.llm_response).calculate(),
            "prompt": self.prompt,
        }
        #print("self.output:",self.output)

        # 检查triplets字段是否存在：如果不存在，记录错误日志并抛出异常。
        #try:
        #    self.output["annotator"]["triplets"]
        #except KeyError:
        #    LOG.error("No triplets found in LLM's output")
        #    LOG.info(f"LLM's output: {self.llm_response.choices[0].message.content}")
        #    raise
        
        # 计算三元组的数量
        self.output["annotator"] = {"triplets": self.output["annotator"]}
        count = len(self.output["annotator"]["triplets"])
        self.output["annotator"]["triples_count"] = count

        return self.output