"""
实体标注对比评估系统
功能：对比BIO标注文件与JSON三元组标注的实体匹配度
修改说明：
1. 修复字典解包语法错误（零宽度空格问题）
2. 优化实体位置匹配逻辑
3. 增强类型安全性
"""
import os
import json
import glob
from collections import defaultdict
from typing import List, Tuple, Dict
import csv

# ========================
# 文件解析模块
# ========================
def parse_bio(bio_path: str) -> List[Tuple[str, str]]:
    """
    解析BIO格式标注文件
    参数：
        bio_path: _bio.txt文件路径（如：2017-2-1_bio.txt）
    返回：
        List[Tuple[实体文本, 实体类型]]（如：[("Ford Fiesta", "Veh")]）
    处理逻辑：
        1. 支持B/I/E/S标签格式（如B_Veh、E_Att）
        2. 自动合并连续实体标记
        3. 处理跨行实体（空行分割段落）
    """
    entities = []
    current_entity = []
    current_type = None
    
    with open(bio_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 段落分隔处理
                if current_entity:
                    entities.append((' '.join(current_entity), current_type))
                    current_entity = []
                    current_type = None
                continue

            # 处理标签格式（兼容B_Veh和B-Veh两种格式）
            token, raw_tag = line.split(' ', 1)
            raw_tag = raw_tag.replace('-', '_')  # 统一分隔符
            prefix = raw_tag[0]
            
            # 提取实体类型（处理复合标签如B_AP_M_1）
            if prefix in ('B', 'S'):
                entity_type = '_'.join(raw_tag.split('_')[1:]) if '_' in raw_tag else raw_tag[2:]
                if current_entity:
                    entities.append((' '.join(current_entity), current_type))
                current_entity = [token]
                current_type = entity_type
            elif prefix in ('I', 'E') and current_type:
                current_entity.append(token)
                if prefix == 'E':
                    entities.append((' '.join(current_entity), current_type))
                    current_entity = []
                    current_type = None
            else:
                if current_entity:
                    entities.append((' '.join(current_entity), current_type))
                current_entity = []
                current_type = None

        # 处理文件末尾的未闭合实体
        if current_entity:
            entities.append((' '.join(current_entity), current_type))
    
    return entities

def parse_json(json_path: str) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    """
    解析JSON三元组文件
    参数：
        json_path: .json文件路径（如：2017-2-1.json）
    返回：
        Dict[Tuple[实体文本, 角色], 位置列表]（如：{("Ford Fiesta", "subject"): [(15, 27)]}）
    处理逻辑：
        1. 支持嵌套triplets结构（兼容列表和字典格式）
        2. 自动定位实体在原文中的位置
        3. 角色类型映射（subject/object → 实体类型）
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_text = data['CTI']['text']  # 保留原始大小写
    text_lower = original_text.lower()   # 用于小写匹配
    triplets = data['IE'].get('triplets', [])
    
    # 处理嵌套结构（兼容不同标注格式）
    if isinstance(triplets, dict):
        triplets = triplets.get('triplets', [])
    
    entity_map = defaultdict(list)
    
    for triplet in triplets:
        for role in ['subject', 'object']:
            entity = triplet[role].strip()
            # 增强定位逻辑：处理前后空格
            clean_entity = ' '.join(entity.split())
            start = text_lower.find(clean_entity.lower())
            if start != -1:
                end = start + len(clean_entity)
                entity_map[(clean_entity, role)].append((start, end))
    
    return entity_map

# ========================
# 评估计算模块
# ========================
def evaluate_pair(bio_entities: List[Tuple[str, str]], 
                 json_entities: Dict[Tuple[str, str], List[Tuple[int, int]]]) -> dict:
    """
    单文件对比评估
    返回：
        包含precision/recall/f1等指标的字典
    """
    # 生成位置映射（兼容部分匹配）
    bio_positions = {}
    for ent_text, ent_type in bio_entities:
        start = ent_text.lower().find(ent_text.lower())
        if start != -1:
            end = start + len(ent_text)
            bio_positions[(start, end)] = ent_type
    
    # 初始化统计指标
    metrics = {
        'tp': 0,    # 完全匹配（位置+类型）
        'fp': 0,    # JSON中存在但BIO中不存在
        'fn': 0,    # BIO中存在但JSON中不存在
        'partial': 0, # 部分位置匹配
        'type_error':0 # 类型不匹配
    }
    
    # 检查JSON中的每个实体
    for (ent_text, role), positions in json_entities.items():
        found = False
        for (start, end) in positions:
            # 精确位置匹配
            if (start, end) in bio_positions:
                metrics['tp'] += 1
                found = True
                # 类型校验（角色首字母需匹配类型）
                if role[0].upper() != bio_positions[(start, end)][0].upper():
                    metrics['type_error'] += 1
                break
            # 部分位置匹配（实体被包含）
            elif any(s <= start and end <= e for (s, e) in bio_positions):
                metrics['partial'] += 1
                found = True
                break
        
        if not found:
            metrics['fp'] += 1
    
    # 统计BIO中未匹配的实体
    matched_bio = set()
    for pos in bio_positions:
        for json_pos in json_entities.values():
            if any(s <= pos[0] and pos[1] <= e for (s,e) in json_pos):
                matched_bio.add(pos)
    metrics['fn'] = len(bio_positions) - len(matched_bio)
    
    # 计算最终指标
    precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0
    recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    result = {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': metrics['tp'],
        'fp': metrics['fp'],
        'fn': metrics['fn'],
        'partial': metrics['partial'],
        'type_error': metrics['type_error']
    }
    return result
# ========================
# 主执行模块
# ========================
def main(json_dir: str, bio_dir: str):
    """
    主执行函数
    新增功能：
        1. 输出每个文件的详细评估结果
        2. 显示表格格式的详细统计
    """
    # 文件匹配
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    file_pairs = []
    for j_path in json_files:
        base_name = os.path.basename(j_path).replace('.json', '')
        b_path = os.path.join(bio_dir, f"{base_name}_bio.txt")
        if os.path.exists(b_path):
            file_pairs.append((j_path, b_path))

    # 存储详细结果
    file_results = []
    all_metrics = []
    
    # 执行批量评估
    for j_path, b_path in file_pairs:
        # 解析文件
        bio_ents = parse_bio(b_path)
        json_ents = parse_json(j_path)
        
        # 评估并存储结果
        metrics = evaluate_pair(bio_ents, json_ents)
        file_name = os.path.basename(j_path)
        file_results.append((file_name, metrics))
        all_metrics.append(metrics)

    # ========================
    # 新增CSV输出功能
    # ========================
    csv_path = os.path.join(os.getcwd(), "evaluation_report.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            '文件名', '精确率', '召回率', 'F1', 
            '完全匹配','标注', '缺失标注',
            '部分匹配', '类型错误'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 写入表头和中文字段
        writer.writeheader()
        
        # 写入每个文件的结果
        for file_name, metrics in file_results:
            writer.writerow({
                '文件名': file_name,
                '精确率': f"{metrics['precision']:.2%}",
                '召回率': f"{metrics['recall']:.2%}",
                'F1': f"{metrics['f1']:.2%}",
                '完全匹配': metrics['tp'],
                '多余标注': metrics['fp'],
                '缺失标注': metrics['fn'],
                '部分匹配': metrics['partial'],
                '类型错误': metrics['type_error']
            })

    # ========================
    # 修改后的控制台输出提示
    # ========================
    print(f"\n{' 详细报告已保存为CSV文件 ':=^80}")
    print(f"文件路径：{csv_path}")
    print(f"包含 {len(file_results)} 个文件的详细评估数据\n")

    
    # ========================
    # 保留原有汇总统计
    # ========================
    # 计算平均值（保留原有逻辑）
    final = {
        'precision': sum(m['precision'] for m in all_metrics) / len(all_metrics),
        'recall': sum(m['recall'] for m in all_metrics) / len(all_metrics),
        'f1': sum(m['f1'] for m in all_metrics) / len(all_metrics),
        'avg_tp': sum(m['tp'] for m in all_metrics) / len(all_metrics),
        'avg_fp': sum(m['fp'] for m in all_metrics) / len(all_metrics),
        'avg_fn': sum(m['fn'] for m in all_metrics) / len(all_metrics)
    }

    # 控制台输出（更新为更清晰的格式）
    print((
    "\n{:=^80}\n"
    "{:^80}\n"
    "{:=^80}\n\n"
    "全局平均指标：\n"
    "{:-^80}\n"
    "精确率 | 召回率 | F1值 | 完全匹配 | 多余标注 | 缺失标注\n"
    "{:.2%} | {:.2%} | {:.2%} | {:^7.1f} | {:^7.1f} | {:^7.1f}\n\n"
).format(
    "",  # 首行分隔符
    " 综合评估报告 ",  # 标题
    "",  # 尾行分隔符
    "",  # 指标分隔符
    final['precision'], 
    final['recall'], 
    final['f1'],
    final['avg_tp'], 
    final['avg_fp'], 
    final['avg_fn']
))
    
    # 可视化（需安装matplotlib）
    try:
        import matplotlib.pyplot as plt
        labels = ['Precision', 'Recall', 'F1']
        values = [final['precision'], final['recall'], final['f1']]
        
        plt.figure(figsize=(8,4))
        plt.bar(labels, values, color=['#2c7bb6','#abd9e9','#fdae61'])
        plt.title('Entity Recognition Performance')
        plt.ylim(0, 1)
        plt.savefig('performance.png')
        print("可视化图表已保存为 performance.png")
    except ImportError:
        print("（提示：安装matplotlib后可生成可视化图表）")

if __name__ == '__main__':
    # 规范化路径格式（防止转义问题）
    JSON_DIR = r"D:/BBBB/graduation project/project/CTI-main1/Eval/RQ4/DemoNum/Automotive(qwen-plus)"
    BIO_DIR = r"D:/BBBB/graduation project/project/CTI-main1/dataset/IE-inputs/Automotive BIOES"
    
    # 执行评估（增加异常处理）
    try:
        main(JSON_DIR, BIO_DIR)
    except Exception as e:
        print(f"评估失败: {str(e)}")
        raise