import json
import os 
from omegaconf import DictConfig, OmegaConf
from jinja2 import Template, Environment, FileSystemLoader, meta
from LLMCaller import LLMCaller
from UsageCalculator import UsageCalculator

# 用于处理语言模型（LLM）的链接任务
class LLMLinker:
    def __init__(self, linker):
        self.config = linker.config
        self.predicted_triples = [] # 用于存储预测的三元组
        self.response_times = []    # 用于存储响应时间
        self.usages = []           # 用于存储使用情况
        self.main_nodes = linker.main_nodes # 主节点
        self.linker = linker     # Linker类的实例
        self.js = linker.js     # JSON文件的内容
        self.inFile = linker.inFile # 输入文件
        self.CTI_Source = linker.CTI_Source # 数据源标识符
        self.topic_node = linker.topic_node # 主题节点

    def link(self):

        # 遍历 self.main_nodes 中的每个主节点，为每个主节点生成提示（prompt）。创建 LLMCaller 对象并调用 LLM，解析 LLM 的响应内容。
        for main_node in self.main_nodes:
            prompt = self.generate_prompt(main_node)
            llmCaller = LLMCaller(self.config, prompt)
            max_regenerate_times = 3
            # for regenerate_times in range(max_regenerate_times):
            self.llm_response, self.response_time = llmCaller.call()
            print("self.llm_response: ", self.llm_response)
            self.usage = UsageCalculator(self.llm_response).calculate()
            self.response_content = json.loads(self.llm_response.choices[0].message.content)
            print("response content: ", self.response_content)
            # print("type of response content: ", type(self.response_content))
            # 尝试从响应内容中提取预测的三元组，如果响应内容的格式不符合预期，使用备用方法提取值。
            try:
                # 分别提取预测的主体（subject）、关系（relation）和客体（object）。
                pred_sub = self.response_content["predicted_triple"]['subject']
                pred_obj = self.response_content["predicted_triple"]['object']
                pred_rel = self.response_content["predicted_triple"]['relation']
                # break
            # 异常处理
            except:
                # 捕获异常并尝试从 self.response_content 的值中直接提取数据。
                values = list(self.response_content.values())
                pred_sub, pred_rel, pred_obj = values[0], values[1], values[2]
                # 尝试直接提取 relation、subject 和 object：
                # try:
                #     pred_rel = self.response_content['relation']
                #     pred_sub = self.response_content['subject']
                #     pred_obj = self.response_content['object']
                #     # break
                # except:
                      # 尝试提取 predicted_subject、predicted_object 和 predicted_relation，如果提取失败，打印错误信息并使用“hallucination”作为预测的三元组。
                #     try:
                #         pred_sub = self.response_content['predicted_subject']
                #         pred_obj = self.response_content['predicted_object']
                #         pred_rel = self.response_content['predicted_relation']
                #     except:
                #         print("Error: The LLM model does not return the predicted triple!")
                #         print(f"Error file name: {self.inFile}")
                #         pred_sub = "hallucination"
                #         pred_obj = "hallucination"
                #         pred_rel = "hallucination"

            # 根据预测的三元组和主节点及主题节点的文本内容，判断是否匹配。如果匹配，则将新的主体和客体节点添加到预测的三元组中。
            if pred_sub == main_node["entity_text"] and pred_obj == self.topic_node["entity_text"]:
                new_sub = {
                    "entity_id": main_node["entity_id"],
                    "mention_text": main_node["entity_text"]
                }
                new_obj = self.topic_node
           
            elif pred_obj == main_node["entity_text"] and pred_sub == self.topic_node["entity_text"]:
                new_sub = self.topic_node
                new_obj = {
                    "entity_id": main_node["entity_id"],
                    "mention_text": main_node["entity_text"]
                }

            # 如果预测的主体和客体与未访问的主体和主题实体不匹配，则打印错误信息并使用“hallucination”作为预测的三元组。
            else:
                print("Error: The predicted subject and object do not match the unvisited subject and topic entity, the LLM produce hallucination!")
                print(f"Hallucinated file name: {self.inFile}")
                new_sub = {
                    "entity_id": "hallucination",
                    "mention_text": "hallucination"
                }
                new_obj = {
                    "entity_id": "hallucination",
                    "mention_text": "hallucination"
                }

            # 构造预测的三元组
            self.predicted_triple = {
                "subject": new_sub,
                "relation": pred_rel,
                "object": new_obj
            }
            # 记录响应时间和资源使用情况
            self.predicted_triples.append(self.predicted_triple)
            self.response_times.append(self.response_time)
            self.usages.append(self.usage)

        # 构造链接结果，包含预测的链接、总响应时间、使用的模型和总资源使用情况
        LP = {
            "predicted_links": self.predicted_triples,
            #response time equals to the sum of all response times
            "response_time": sum(self.response_times),
            "model": self.config.model,
            #usage equals to the sum of all usages
            "usage": {
                "input": {
                    "tokens": sum([usage["input"]["tokens"] for usage in self.usages]),
                    "cost": sum([usage["input"]["cost"] for usage in self.usages])
                },
                "output": {
                    "tokens": sum([usage["output"]["tokens"] for usage in self.usages]),
                    "cost": sum([usage["output"]["cost"] for usage in self.usages])
                },
                "total": {
                    "tokens": sum([usage["total"]["tokens"] for usage in self.usages]),
                    "cost": sum([usage["total"]["cost"] for usage in self.usages])
                }
            }
        }
        return LP

    # 生成提示方法：generate_prompt
    def generate_prompt(self, main_node):
            # 初始化 Jinja2 环境，FileSystemLoader指定模板文件所在的目录
            env = Environment(loader=FileSystemLoader(self.config.link_prompt_folder))
            # 使用 env.loader.get_source 获取模板文件的内容，使用 env.parse 解析模板内容
            parsed_template = env.parse(env.loader.get_source(env, self.config.link_prompt_file)[0])
            #  加载指定的模板文件
            template = env.get_template(self.config.link_prompt_file)
            # 查找模板中未声明的变量
            variables = meta.find_undeclared_variables(parsed_template)

            # 如果模板中包含变量（variables is not {}），使用提供的变量值渲染模板，否则直接渲染模板
            if variables is not {}: # if template has variables
                    User_prompt = template.render(main_node=main_node["entity_text"], CTI=self.js["CTI"]["text"], topic_node=self.topic_node["entity_text"])
            else:
                User_prompt = template.render()
            # 构造提示
            prompt = [{"role": "user", "content": User_prompt}] # role：用户角色，content：提示内容

            # 构造子文件夹路径
            subFolderPath = os.path.join(self.config.link_prompt_set, self.CTI_Source)
            # 使用 os.makedirs 确保子文件夹存在，exist_ok=True 表示如果文件夹已存在，不会抛出异常。
            os.makedirs(subFolderPath, exist_ok=True)
            # 保存提示文件
            with open(os.path.join(subFolderPath, self.inFile.split('.')[0] + ".txt"), 'w') as f:
                f.write(json.dumps(User_prompt, indent=4).replace("\\n", "\n").replace('\\"', '\"'))
            return prompt

        
    # def find_topic_entity(self):
    #     attack_malware_dict = {}
    #     for triple in self.js["EA"]["aligned_triplets"]:
    #          for key, node in triple.items():
    #              if key in ["subject", "object"]:
    #                  if node["mention_class"] == "Attacker" or node["mention_class"] == "Malware":
    #                     #check if the entity is in the attack_malware_dict
    #                     if node["entity_id"] in attack_malware_dict:
    #                         continue
    #                     else:
    #                         attack_malware_dict[node["entity_id"]] = {
    #                             "entity_text": node["mention_text"],
    #                             "mentions_time": len(node["mentions_merged"])
    #                         }
    #     #return the entity_id with the most mentions
    #     max_entity_id = max(attack_malware_dict, key=lambda x: attack_malware_dict[x]["mentions_time"])
    #     max_entity_text = attack_malware_dict[max_entity_id]["entity_text"]
    #     return {
    #         "entity_id": max_entity_id,
    #         "entity_text": max_entity_text
    #     }
    