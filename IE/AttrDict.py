class AttrDict:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = AttrDict(**value)  # 递归处理嵌套字典
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({vars(self)})"