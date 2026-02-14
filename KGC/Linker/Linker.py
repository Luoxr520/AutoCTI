import json
import os 
from omegaconf import DictConfig, OmegaConf
from LLMLinker import LLMLinker
from collections import defaultdict # 用于创建默认字典，方便统计和处理

# 用于处理图结构中的三元组（triplets），并找到每个子图（subgraph）的主节点（main node）。
class Linker:
    def __init__(self, config: DictConfig, CTI_Source, inFile):
        self.config = config
        self.CTI_Source = CTI_Source
        self.inFile = inFile

        # 读取输入文件,提取 aligned_triplets，即对齐的三元组列表。
        infile_path = os.path.join(self.config.inSet, self.CTI_Source, self.inFile)
        with open(infile_path, 'r') as f:
            self.js = json.load(f)
            self.aligned_triplets = self.js["EA"]["aligned_triplets"]
            

        # print(f"aligned_triplets: {self.aligned_triplets}")
        # 构建一个图的邻接列表来表示这些三元组的连接
        self.graph = {}

        # 填充图
        # 遍历 aligned_triplets，提取每个三元组的主体和对象的实体 ID
        for triplet in self.aligned_triplets:
            subject_entity_id = triplet["subject"]["entity_id"]
            object_entity_id = triplet["object"]["entity_id"]
            
            if subject_entity_id not in self.graph:
                self.graph[subject_entity_id] = []
            if object_entity_id not in self.graph:
                self.graph[object_entity_id] = []
            
            # 构建无向图，将每个主体和对象的实体 ID 添加到邻接列表中
            self.graph[subject_entity_id].append(object_entity_id)
            self.graph[object_entity_id].append(subject_entity_id)

        # 调用 find_disconnected_subgraphs 方法，找到图中所有不连通的子图。
        self.subgraphs = self.find_disconnected_subgraphs()
        self.main_nodes = []
        # # 使用DFS来检查连通性
        # self.visited = set()

        # 对于每个子图，找到主节点
        for i, subgraph in enumerate(self.subgraphs):
            main_node_entity_id = self.get_main_node(subgraph)
            main_node = self.get_node(main_node_entity_id)
            print(f"subgraph {i}: main node: {main_node['entity_text']}")
            self.main_nodes.append(main_node)

        # 从第一个节点开始DFS
        self.dfs(next(iter(self.graph)))

        # 检查是否访问了所有节点
        is_connected = len(self.visited) == len(self.graph)
        print(f"Is the graph connected? {is_connected}")

        # 如果图不是连通的，对于未访问的子图，再使用dfs
        if not is_connected:
            for node in self.graph.keys():
                if node not in self.visited:
                    self.dfs(node)

        unvisited_nodes = set(self.graph.keys()) - self.visited
        print(f"Unvisited nodes: {unvisited_nodes}")

        unvisted_subjects = {}
        for triplet in self.aligned_triplets:
            if triplet["subject"]["entity_id"] in unvisited_nodes:
                unvisted_subjects[triplet["subject"]["entity_id"]] = triplet["subject"]

        # 找到主题节点（topic node）
        self.topic_node = self.get_topic_node(self.subgraphs)
        # 从主节点列表中移除主题节点
        self.main_nodes = [node for node in self.main_nodes if node["entity_id"] != self.topic_node["entity_id"]]
        # 创建 LLMLinker 对象并调用其 link 方法，获取链接结果，将主题节点、主节点、子图信息和子图数量添加到 self.js["LP"] 中
        self.js["LP"] = LLMLinker(self).link()
        self.js["LP"]["topic_node"] = self.topic_node
        self.js["LP"]["main_nodes"] = self.main_nodes
        self.js["LP"]["subgraphs"] = [list(subgraph) for subgraph in self.subgraphs]
        self.js["LP"]["subgraph_num"] = len(self.subgraphs)
        # 构造输出文件夹路径，并确保输出文件夹存在
        outfolder = os.path.join(self.config.outSet, self.CTI_Source)
        os.makedirs(outfolder, exist_ok=True)
        outfile_path = os.path.join(outfolder, self.inFile)

        with open(outfile_path, 'w') as f:
            json.dump(self.js, f, indent=4)

    # 深度优先搜索
    def dfs(self, node):
        if node in self.visited:
            return
        self.visited.add(node)
        for neighbour in self.graph[node]:
            self.dfs(neighbour)

    # 找到不连通的子图
    def find_disconnected_subgraphs(self):
        self.visited = set()
        subgraphs = []

        for start_node in self.graph.keys():
            if start_node not in self.visited:
                # 每次找到一个新的子图时，清空当前的访问记录，但只记录这个子图的访问
                current_subgraph = set()
                self.dfs_collect(start_node, current_subgraph)
                subgraphs.append(current_subgraph)

        return subgraphs

    #  深度优先搜索
    def dfs_collect(self, node, current_subgraph):
        if node in self.visited:
            return
        self.visited.add(node)
        current_subgraph.add(node)  # 添加节点到当前子图
        for neighbour in self.graph[node]:
            self.dfs_collect(neighbour, current_subgraph)

    # 找到每个子图的主节点
    def get_main_node(self, subgraph):
        # 统计每个节点的出度
        outdegrees = defaultdict(int)
        self.directed_graph = {}
        #build directed graph
        for triplet in self.aligned_triplets:
            subject_entity_id = triplet["subject"]["entity_id"]
            object_entity_id = triplet["object"]["entity_id"]
            if subject_entity_id not in self.directed_graph:
                self.directed_graph[subject_entity_id] = []
            self.directed_graph[subject_entity_id].append(object_entity_id)
            outdegrees[subject_entity_id] += 1
            outdegrees[object_entity_id] += 1
        #找到出度最大的节点
        max_outdegree = 0
        main_node = None
        for node in subgraph:
            if outdegrees[node] > max_outdegree:
                max_outdegree = outdegrees[node]
                main_node = node
        return main_node
    
    # 获取节点信息
    def get_node(self, entity_id):
        for triplet in self.aligned_triplets:
            for key, node in triplet.items():
                if key in ["subject", "object"]:
                    if node["entity_id"] == entity_id:
                        return node
                    
    # 找到主题节点
    def get_topic_node(self, subgraphs):
        #节点数最多的子图为主节点所在图
        max_node_num = 0
        for subgraph in subgraphs:
            if len(subgraph) > max_node_num:
                max_node_num = len(subgraph)
                main_subgraph = subgraph
        #找到最大图的主节点
        return self.get_node(self.get_main_node(main_subgraph))