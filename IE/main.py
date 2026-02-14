from LLMAnnotator import LLMAnnotator
import os 
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm # 进度条
'''
@hydra.main：这是hydra框架的装饰器，用于指定配置文件的路径和名称。
config_path="config"：配置文件所在的目录。
config_name="demoNum=4"：配置文件的名称。hydra会加载config/demoNum=4.yaml文件。
version_base="1.2"：指定hydra的版本。
config: DictConfig：函数参数config是一个DictConfig对象，它包含了从配置文件中加载的配置数据。'''
@hydra.main(config_path="config", config_name="demoNum=5", version_base="1.2")
def run(config: DictConfig):
        #print(OmegaConf.to_yaml(config))
        if hasattr(config, 'inFile'):
            LLMAnnotator(config, config.CTI_Source, config.inFile).annotate()
            # 如果配置文件中存在inFile属性，说明需要处理单个文件,创建一个LLMAnnotator对象，并调用其annotate方法进行注释处理。
            print("inFile处理完成")
        elif hasattr(config, 'CTI_Source'):
            inFolder = os.path.join(config.inSet, config.CTI_Source)
            for JSONFile in tqdm(os.listdir(inFolder), desc="处理文件夹中的文件"):
                LLMAnnotator(config, config.CTI_Source, JSONFile).annotate()
            print("CTI_Source处理完成")
            # 如果配置文件中存在CTI_Source属性，说明需要处理一个文件夹中的所有文件,构造文件夹路径inFolder,遍历该文件夹中文件,对每个文件，创建一个LLMAnnotator对象并调用annotate方法进行注释处理。
        else:
            for CTI_Source in os.listdir(config.inSet):
                annotatedCTICource = [dir for dir in os.listdir(config.outSet)]
                if CTI_Source in annotatedCTICource:
                    continue
                if CTI_Source.endswith('.json'):
                    continue
                # 如果配置文件中既没有inFile也没有CTI_Source，则遍历config.inSet目录中的所有子目录。如果子目录以.json结尾，跳过
                FolderPath = os.path.join(config.inSet, CTI_Source)
                for JSONFile in tqdm(os.listdir(FolderPath), desc=f"处理 {CTI_Source} 文件夹中的文件"):
                    LLMAnnotator(config, CTI_Source, JSONFile).annotate() 
if __name__ == "__main__":
    run()
     
