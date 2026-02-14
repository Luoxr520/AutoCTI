from promptConstructor import PromptConstructor # 构建 LLM 的输入提示
from demoRetriever import DemoRetriever # 从 CTI 中检索示例
from instructionLoader import InstructionLoader # 加载任务指令
from LLMcaller import LLMCaller # 调用 LLM 模型
from responseParser import ResponseParser # 解析 LLM 的输出
import json
import os 
from FalseCTIDetector import FalseCTIDetector
from knowledge_graph_builder import KnowledgeGraphBuilder


class LLMAnnotator:
    def __init__(self, config, CTI_Source, inFile):
        self.config = config
        self.CTI_Source = CTI_Source
        self.inFile = inFile
        
        self.kg_builder = KnowledgeGraphBuilder() # 初始化知识图谱构建器
    
    def load_cti_data(self):
        inFolder = os.path.join(self.config.inSet, self.CTI_Source)
        with open(os.path.join(inFolder, self.inFile), 'r') as f:
            return json.load(f)
    
    # 定义注释方法，负责整个注释流程
    def annotate(self):
        self.inFileJSON = self.load_cti_data()
        print("CTI 数据读取文件成功")

        # 检索示例
        if self.config.retriever == "fixed":
            self.demos = ""
        else:
            self.demos, self.demosInfo = DemoRetriever(self).retriveDemo()

        # 构造提示
        # 获取输入文件的文件名（去掉扩展名）
        self.inFilename = os.path.splitext(self.inFile)[0]
        # 调用 PromptConstructor 类的 generate_prompt() 方法，根据配置和输入数据生成 LLM 的输入提示
        self.prompt = PromptConstructor(self).generate_prompt()
        print("LLMAnnotator.py调用prompt成功")

        # 调用 LLMCaller 类的 call() 方法，调用 LLM 模型
        self.llm_response, self.response_time, self.JSONResp = LLMCaller(self.config, self.prompt).call()
        print("LLMAnnotator.py调用LLMCaller成功")
        #print("self.llm_response:",self.llm_response)
        #print("self.response_time:",self.response_time)
        #print("self.JSONResp:",self.JSONResp)
        # 解析 LLM 的输出
        self.output = ResponseParser(self).parse()
        print("LLMAnnotator.py解析LLM输出成功")

        # 构建知识图谱
        print("self.output[\"annotator\"][\"triplets\"]:", self.output["annotator"]["triplets"])
        print("type:", type(self.output["annotator"]["triplets"]))

        # 直接使用三元组列表
        triplets_list = self.output["annotator"]["triplets"]

        # 处理各种格式的三元组
        for triplet_item in triplets_list:
            # 调试打印
            print(f"Processing triplet item: {triplet_item} (type: {type(triplet_item)})")
            
            # 处理不同类型的数据
            if isinstance(triplet_item, dict):
                # 字典格式处理
                subject = triplet_item.get("subject", "")
                relation = triplet_item.get("relation", "")
                obj = triplet_item.get("object", "")
            elif isinstance(triplet_item, str):
                # 字符串格式处理
                try:
                    # 尝试解析JSON字符串
                    triplet_dict = json.loads(triplet_item)
                    subject = triplet_dict.get("subject", "")
                    relation = triplet_dict.get("relation", "")
                    obj = triplet_dict.get("object", "")
                except json.JSONDecodeError:
                    # 如果解析失败，尝试简单拆分
                    parts = triplet_item.split("|")
                    subject = parts[0] if len(parts) > 0 else ""
                    relation = parts[1] if len(parts) > 1 else ""
                    obj = parts[2] if len(parts) > 2 else ""
            else:
                # 其他类型跳过
                print(f"Skipping unsupported triplet type: {type(triplet_item)}")
                continue
            
            # 添加到知识图谱
            if subject and relation and obj:
                triplet = (subject, relation, obj)
                self.kg_builder.add_triplet(triplet)
                print(f"Added triplet: {triplet}")
            else:
                print(f"Skipping incomplete triplet: subject={subject}, relation={relation}, object={obj}")

        # 保存和可视化知识图谱
        self.kg_builder.save_graph(os.path.join(self.config.outSet, "knowledge_graph.pkl"))
        # self.save_results()



        # 提取提示 ID (从 LLM 响应中提取 ID 的最后三位)和模板名称
        self.promptID = self.llm_response.id[-3:]
        templ = self.config.templ.split('.')[0]

        # 保存生成的提示到文件(.jinja)
        self.outPromptFolder = os.path.join(self.config.ie_prompt_store, self.CTI_Source)
        os.makedirs(self.outPromptFolder, exist_ok=True)

        self.outPromptFile = os.path.join(self.outPromptFolder, f'{self.inFilename}-{templ}-{self.promptID}.jinja')

        with open(self.outPromptFile, 'w') as f:
            json_str = json.dumps(self.output["prompt"])
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)
        
        # 保存注释结果到文件(.json)
        self.outFolder = os.path.join(self.config.outSet, self.CTI_Source)
        os.makedirs(self.outFolder, exist_ok=True)
        
        self.outFile = os.path.join(self.outFolder, f'{self.inFilename}.json')

        # 构造输出 JSON 文件内容并保存
        with open(self.outFile, 'w') as f:
            outJSON = {}
            ###CTI
            outJSON["CTI"] = {}
            outJSON["CTI"]["text"] = self.output["CTI"]
            outJSON["CTI"]["link"] = self.output["link"]
            ###IE results
            outJSON["IE"] = {}
            outJSON["IE"]["triplets"] = self.output["annotator"]["triplets"]
            outJSON["IE"]["triples_count"] = self.output["annotator"]["triples_count"]
            outJSON["IE"]["cost"] = self.output["usage"]
            outJSON["IE"]["time"] = self.response_time
            ###Prompt
            outJSON["IE"]["Prompt"] = {}
            outJSON["IE"]["Prompt"]["constructed_prompt"] = self.outPromptFile
            outJSON["IE"]["Prompt"]["prompt_template"] = self.config.templ
            outJSON["IE"]["Prompt"]["demo_retriever"]= self.config.retriever.type
            outJSON["IE"]["Prompt"]["demo_number"] = self.config.shot
            outJSON["IE"]["Prompt"]["demos"] = self.demosInfo
            if self.config.retriever.type == "kNN":
                outJSON["IE"]["Prompt"]["permutation"] = self.config.retriever.permutation
            #write to file
            json.dump(outJSON, f, indent=4)