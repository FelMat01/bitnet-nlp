class DatasetGenerator:
    def __init__(self, prompt_template) -> None:
        self.prompt_template =  prompt_template
    
    def generate(self, context:str, classes:list[str], samples_per_class:int = 5, number_of_words:int = 50) -> dict[str,str]:
        pass