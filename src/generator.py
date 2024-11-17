from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """
        <|system|>
        Generate a synthetic data example for each of the following classes based on the context.
        Please provide one example for each class in JSON format.
        Format the output as:
        {{ class1 : example1, class2: example2}}
        
        </s>
        <|user|>
        Context:
        {context}
        
        Classes: 
        {classes}

        </s>
        <|assistant|>
        """

class DatasetGenerator:
    def __init__(self,
        model_repo: str,
        classes: list[str],
        general_context: str = None):
        
        self.classes = classes
        self.general_context = general_context
        
        self.llm = HuggingFaceHub(repo_id=model_repo, model_kwargs={"temperature": 0.5, "max_length": 64,"max_new_tokens":512})
         
    def generate(self):
        
        context = "This classifier is about distinguishing types of beverages based on brands. Generate 3 examples for each class."

        prompt = PROMPT_TEMPLATE.format(context=context, classes=self.classes)

        response = self.llm.predict(prompt)
        print(response)
    
generator = DatasetGenerator(model_repo = "huggingfaceh4/zephyr-7b-alpha", classes= ["coffee", "tea", "soda"])

generator.generate()