import os 
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from LLMTagger import LLMTagger


# 对文件进行标记（tagging）
@hydra.main(config_path="config", config_name="qwen+1shot", version_base="1.2")
def run(config: DictConfig):
    # 处理单个文件
    if hasattr(config, "inFile"): # tag a single file
        LLMTagger(config).tag(config.CTI_source, config.inFile)

    # 处理一个文件夹中的文件
    elif hasattr(config, "CTI_Source"): # tag all files in a single CTI_source
        inFolder = os.path.join(config.inSet, config.CTI_Source)
        for file in os.listdir(inFolder):
            LLMTagger(config).tag(config.CTI_source, file)

    # 处理所有文件夹中的文件
    else: # tag all files in all CTI_sources
        for CTI_source in os.listdir(config.inSet):
            if CTI_source == "prompt_store":
                continue
            annotatedCTICource = [dir for dir in os.listdir(config.outSet)]
            if CTI_source in annotatedCTICource:
                continue
            inFolder = os.path.join(config.inSet, CTI_source)
            for file in os.listdir(inFolder):
                LLMTagger(config).tag(CTI_source, file)

if __name__ == "__main__":
    run()