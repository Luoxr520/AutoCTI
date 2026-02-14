import networkx as nx
from typing import List, Dict, Tuple, Any
import pickle

class KnowledgeGraphBuilder:
    """知识图谱基类"""
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_index = {}

    def add_triplet(self, triplet: Tuple[str, str, str]):
        """添加三元组"""
        subject, predicate, obj = triplet
        self.graph.add_node(subject, type='entity')
        self.graph.add_node(obj, type='entity')
        self.graph.add_edge(subject, obj, label=predicate)
        
    def save_graph(self, output_path):
        """使用标准 Pickle 保存"""
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f)

    def load_graph(self, input_path):
        with open(input_path, 'rb') as f:
            self.graph = pickle.load(f)

    

class EnhancedKnowledgeGraph(KnowledgeGraphBuilder):
    """增强型知识图谱"""
    def find_related_nodes(self, entity: str, depth=2) -> List[str]:
        """查找关联节点"""
        return list(nx.ego_graph(self.graph, entity, radius=depth).nodes())

    def semantic_search(self, keywords: List[str]) -> Dict[str, List[str]]:
        """语义搜索"""
        results = {}
        for kw in keywords:
            results[kw] = [n for n in self.graph.nodes() if kw.lower() in n.lower()]
        return results

    def get_visualization_elements(self) -> List[Dict[str, Any]]:
        """生成Cytoscape元素"""
        elements = []
        for node in self.graph.nodes():
            elements.append({
                'data': {'id': node, 'label': node},
                'classes': 'entity-node'
            })
        for edge in self.graph.edges(data=True):
            elements.append({
                'data': {
                    'source': edge[0],
                    'target': edge[1],
                    'label': edge[2]['label']
                },
                'classes': 'relation-edge'
            })
        return elements