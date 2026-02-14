import re
import json
from datetime import datetime, timedelta

class FalseCTIDetector:
    def __init__(self, original_cti):
        self.original_cti = original_cti
        self.false_cti = self.generate_false_cti()
        self.rules = self.define_rules()

    def generate_false_cti(self):
        """基于真实 CTI 构建虚假 CTI"""
        false_cti = self.original_cti.copy()
        #print("false_cti[CTI]:",false_cti["CTI"])
        #print("type(false_cti[CTI]):",type(false_cti["CTI"]))
        
        # 修改受害者信息
        false_cti["CTI"] = re.sub(
            r"4LEAF", "Fake Corp", false_cti["CTI"]
        )
        false_cti["CTI"] = re.sub(
            r"Park-Rite", "Nonexistent Tech", false_cti["CTI"]
        )
        false_cti["CTI"] = re.sub(
            r"Family Day Care Services", "Imaginary Retail", false_cti["CTI"]
        )

        # 修改国家信息
        false_cti["CTI"]= re.sub(
            r"American", "Mordanian", false_cti["CTI"]
        )
        false_cti["CTI"] = re.sub(
            r"U\.S\.-based", "Ozzian", false_cti["CTI"]
        )
        false_cti["CTI"] = re.sub(
            r"Canadian", "Neverlandish", false_cti["CTI"]
        )

        # 修改时间线
        false_cti["CTI"] = re.sub(
            r"August 28, 2017", "September 15, 2023", false_cti["CTI"]
        )

        # 添加虚假技术细节
        false_cti["CTI"] = re.sub(
            r"\.akira", ".fake", false_cti["CTI"]
        )
        false_cti["CTI"] = re.sub(
            r"WordPress sites", "Magic Beans Distribution", false_cti["CTI"]
        )

        # 修改威胁行为体信息
        false_cti["CTI"] = re.sub(
            r"Akira", "FakeRansomer", false_cti["CTI"]
        )
        false_cti["CTI"] = re.sub(
            r"Akira's leak site", "http://fakesite.example", false_cti["CTI"]
        )

        return false_cti

    def define_rules(self):
        """定义虚假信息识别规则"""
        return [
            {
                "pattern": r"(Mordanian|Ozzian|Neverlandish)",
                "description": "检测到不存在的国家描述"
            },
            {
                "pattern": r"\.fake",
                "description": "检测到虚假的文件扩展名"
            },
            {
                "pattern": r"Magic Beans Distribution",
                "description": "检测到不现实的传播方式"
            },
            {
                "pattern": r"FakeRansomer",
                "description": "检测到虚构的威胁行为体名称"
            },
            {
                "pattern": r"http://fakesite\.example",
                "description": "检测到虚假的泄露网站地址"
            },
            {
                "pattern": r"September 15, 2023",
                "description": "检测到被篡改的时间信息"
            },
            {
                "pattern": r"Fake Corp|Nonexistent Tech|Imaginary Retail",
                "description": "检测到虚假的受害者信息"
            }
        ]

    def detect_false_info(self):
        """检测虚假 CTI"""
        for rule in self.rules:
            if re.search(rule["pattern"], self.false_cti["CTI"]):
                print(f"检测到虚假信息: {rule['description']}")
                return True
        return False

    def filter_false_info(self):
        """过滤虚假信息并还原真实 CTI"""
        filtered_text = self.false_cti["CTI"]

        # 应用规则进行过滤
        for rule in self.rules:
            filtered_text = re.sub(rule["pattern"], "", filtered_text)

        # 还原真实 CTI 格式
        filtered_cti = self.false_cti.copy()
        filtered_cti["CTI"] = filtered_text
        return filtered_cti

    def run_pipeline(self):
        """运行完整流程"""
        print("原始真实 CTI:")
        print(json.dumps(self.original_cti["CTI"], indent=2))

        print("\n生成的虚假 CTI:")
        print(json.dumps(self.false_cti["CTI"], indent=2))

        print("\n检测虚假信息:")
        if self.detect_false_info():
            print("检测到虚假信息，开始过滤...")

            filtered_cti = self.filter_false_info()
            print("\n过滤后还原的真实 CTI:")
            print(json.dumps(filtered_cti["CTI"], indent=2))
            return filtered_cti
        else:
            print("未检测到虚假信息")
            return self.original_cti