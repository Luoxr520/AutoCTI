import json

# 从一个 JSON 文件中加载指定 ID 的指令
class InstructionLoader:
    # 定义类的初始化方法
    def __init__(self, InstructionID):
        #InstructionID：一个标识符，用于指定需要加载的指令的 ID
        self.instructionID = InstructionID # 将传入的 InstructionID 保存为类的属性
        self.instructionfile = "Toolbox/instruction/instruction.json" # 指令文件的路径
        with open(self.instructionfile, "r") as f:
            data = json.load(f) # 读取指令文件
        self.instruction= data[self.instructionID] # 从指令文件中读取指令
