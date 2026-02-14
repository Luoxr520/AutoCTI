import os 
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from Linker import Linker # 从 Linker 模块导入的类，用于处理文件或文件夹。
from tqdm import tqdm

@hydra.main(config_path="config", config_name="3shot-qwen", version_base="1.2")
def run(config: DictConfig):
    # linker = Linker(config)
    # 处理 addition 配置项,遍历每个 CTI_Source 和对应的文件，调用 Linker 类进行处理。
    if hasattr(config, 'addition'):
        for CTI_Source in config.addition:
            for file in tqdm(config.addition[CTI_Source], desc=f"处理 {CTI_Source} 文件夹中的文件"):
                Linker(config, CTI_Source, file)

    # 处理单个输入文件
    elif hasattr(config, 'inFile'): # If specifiy a input file
        Linker(config, config.CTI_Source, config.inFile)

    # 处理一个文件夹中的所有文件
    elif hasattr(config, 'CTI_Source'): # If specify a CTI_Source
        FolderPath = os.path.join(config.inSet, config.CTI_Source)
        annotated_files = os.listdir(os.path.join(config.outSet, config.CTI_Source))
        for JSONFile in tqdm(os.listdir(FolderPath), desc=f"处理 {config.CTI_Source} 文件夹中的文件"):
            if JSONFile in annotated_files:
                continue
            Linker(config, config.CTI_Source, JSONFile)

    # 遍历所有子目录
    else: # If not specify any input file or CTI_Source
        annotated_CTI_Sources = os.listdir(config.outSet)
        for CTI_Source in os.listdir(config.inSet):
            if CTI_Source in annotated_CTI_Sources:
                continue
            FolderPath = os.path.join(config.inSet, CTI_Source)
            for JSONFile in tqdm(os.listdir(FolderPath), desc=f"处理 {CTI_Source} 文件夹中的文件"):
                Linker(config, CTI_Source, JSONFile)


if __name__ == "__main__":
    run()