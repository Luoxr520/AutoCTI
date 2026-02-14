from typing import List, Dict, Tuple, Any
from knowledge_graph_builder import EnhancedKnowledgeGraph

class KGQAProcessor:
    def __init__(self, kg_path: str):
        self.kg = EnhancedKnowledgeGraph()
        self.kg.load_graph(kg_path)
        
    def process_question(self, question: str) -> Tuple[str, List[str]]:
        """处理问答流程"""
        keywords = self._extract_keywords(question)
        context = self.kg.semantic_search(keywords)
        prompt = self._build_prompt(question, context)
        response = self._call_llm(prompt)
        return self._parse_response(response)

    def _extract_keywords(self, text: str) -> List[str]:
        """关键词提取（示例实现）"""
        return ["attack", "vulnerability"]

    def _build_prompt(self, question: str, context: Dict) -> str:
        """构建LLM提示"""
        return f"Context: {context}\nQuestion: {question}\nAnswer:"

    def _call_llm(self, prompt: str) -> str:
        """调用LLM（示例实现）"""
        return "示例回答：检测到利用CVE-2023-1234的SQL注入攻击，影响Apache服务器2.4版本"

    def _parse_response(self, response: str) -> Tuple[str, List[str]]:
        """解析响应"""
        entities = ["CVE-2023-1234", "Apache Server 2.4"]
        return response, entities