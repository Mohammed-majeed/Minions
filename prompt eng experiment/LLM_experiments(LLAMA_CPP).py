import xml.etree.ElementTree as ET
from llama_cpp import Llama
import csv
import os

class BehaviorTreeGenerator:
    def __init__(self,  
                 model_file, 
                 context_length,
                 gpu_layers,
                 n_threads,
                 temperature,
                 top_p,
                 top_k,
                 repeat_penalty,  
                 seed               
                 ):

        self.error_count = 0

        self.model_file=model_file
        self.context_length=context_length
        self.gpu_layers=gpu_layers
        self.n_threads=n_threads
        self.temperature=temperature
        self.top_p=top_p
        self.top_k=top_k
        self.repeat_penalty=repeat_penalty
        self.echo=False
        self.seed = seed

        self.llm = Llama(
            model_path=self.model_file,
            n_ctx=self.context_length,
            n_gpu_layers=self.gpu_layers,
            n_threads=self.n_threads,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repeat_penalty=self.repeat_penalty,
            echo=self.echo,
            seed=self.seed
            )

    def generate_behavior_tree(self, prompt, fix_mistake):

        
        
        BT_1 = """
        <BehaviorTree>
            <Sequence>
                <Selector>
                    <Sequence>
                        <Condition>behavior</Condition>
                        <Sequence>
                            <Condition>behavior</Condition>
                            <Action>behavior</Action>
                        </Sequence>
                        <Sequence>
                            <Condition>behavior</Condition>
                            <Action>behavior</Action>
                        </Sequence>
                    </Sequence>
                    <Action>behavior</Action>
                </Selector>
                <Selector>
                    <Sequence>
                        <Condition>behavior</Condition>
                        <Selector>
                            <Sequence>
                                <Condition>behavior</Condition>
                                <Action>behavior</Action>
                            </Sequence>
                            <Sequence>
                                <Condition>behavior</Condition>
                                <Action>behavior</Action>
                            </Sequence>
                        </Selector>
                    </Sequence>
                    <Action>behavior</Action>
                </Selector>
            </Sequence>
        </BehaviorTree>
        """

        BT_2 = """
        <BehaviorTree>
            <Sequence>
                <Action>behavior</Action>
            </Sequence>
        </BehaviorTree>
        """
        


        xml_examples = f'''
        <<SYS>> SYSTEM: You are a helpful, respectful, and honest AI assistant. Your task is to generate well-structured XML code for behavior trees based on the provided instructions.<</SYS>>
        [INST] INSTRUCTIONS: You're given examples of behavior tree structures, like:
        1. Example 1:
        {BT_1}
        2. Example 2:
        {BT_2}
        Note that while these examples can guide you, feel free to devise different structures as needed. Your objective is to use the following available behaviors: {self.call_behaviors()}, to craft a behavior tree in XML format. This tree should cater to simulated swarm robots based on the user's command below.
        USER COMMAND: generate behavior tree to "{prompt}". Output only the XML behavior tree without extra text.[/INST]
        RESPONSE: {fix_mistake}
        '''


        BT_3 = """
        <BehaviorTree>
            <Sequence>
                <Selector>
                    <Sequence>
                        <Condition>is_line_formed</Condition>
                        <Action>change_color_to_green</Action>
                        <Action>agent_movement_freeze</Action>
                        <Action>task_completed</Action>
                    </Sequence>
                    <Action>form_line</Action>
                </Selector>
            </Sequence>
        </BehaviorTree>
                """
        BT_4 = """
        <BehaviorTree>
            <Sequence>
                <Selector>
                    <Sequence>
                        <Condition>is_target_detected</Condition>
                        <Action>change_color_to_green</Action>
                        <Action>task_completed</Action>
                    </Sequence>
                    <Action>wander</Action>
                </Selector>                
                <Sequence>
                    <Condition>is_agent_in_nest</Condition>
                    <Action>change_color_to_white</Action>
                </Sequence>
            </Sequence>
        </BehaviorTree>
            """
        

   
        
        two_examples = f'''
        <s> <<SYS>>You are a helpful, respectful, and honest AI assistant. Your task is to generate well-structured XML code for behavior trees based on the provided instructions.<</SYS>>
        [INST] 
        INSTRUCTIONS: Use only the following behaviors: {self.call_behaviors()} to construct behavior tree in XML format to the following command.
        USER COMMAND: generate behavior tree to "Form a line, change color to green, and then freeze movement."[/INST]
        RESPONSE: {BT_3}
        </s>

        <s> <<SYS>>You are a helpful, respectful, and honest AI assistant. Your task is to generate well-structured XML code for behavior trees based on the provided instructions.<</SYS>>
        [INST]
        INSTRUCTIONS: Use only the following behaviors: {self.call_behaviors()} to construct behavior tree in XML format to the following command.
        USER COMMAND: generate behavior tree to "Find food and change color to green, then return to the nest, change color to white."[/INST]
        RESPONSE: {BT_4}
        </s>
        '''

        two_shot_learning = f'''        
        {two_examples}
        <s> <<SYS>>You are a helpful, respectful, and honest AI assistant. Your task is to generate well-structured XML code for behavior trees based on the provided instructions.<</SYS>>
        [INST]
        INSTRUCTIONS: Use only the following behaviors: {self.call_behaviors()} to construct behavior tree in XML format to the following command.
        USER COMMAND: generate behavior tree to "{prompt}". Output only the XML behavior tree without extra text.[/INST]
        RESPONSE: {fix_mistake}
        '''

        one_example = f'''<s> <<SYS>>You are a helpful, respectful, and honest AI assistant. Your task is to generate well-structured XML code for behavior trees based on the provided instructions.<</SYS>>
        [INST]
        INSTRUCTIONS: Use only the following behaviors: {self.call_behaviors()} to construct behavior tree in XML format to the following command.
        USER COMMAND: generate behavior tree to "Find food and change color to green, then return to the nest, change color to white."[/INST]
        RESPONSE: {BT_4}
        </s>
        '''

        one_shot_learning = f'''        
        {one_example}
        <s> <<SYS>>You are a helpful, respectful, and honest AI assistant. Your task is to generate well-structured XML code for behavior trees based on the provided instructions.<</SYS>>
        [INST]
        INSTRUCTIONS: Use only the following behaviors: {self.call_behaviors()} to construct behavior tree in XML format to the following command.
        USER COMMAND: generate behavior tree to "{prompt}". Output only the XML behavior tree without extra text.[/INST]
        RESPONSE: {fix_mistake}
        '''     

        zero_shot_learning = f'''
        <s> <<SYS>>You are a helpful, respectful, and honest AI assistant. Your task is to generate well-structured XML code for behavior trees based on the provided instructions.<</SYS>>
        [INST]
        INSTRUCTIONS: Use only the following behaviors: {self.call_behaviors()} to construct behavior tree in XML format to the following command.
        USER COMMAND: generate behavior tree to "{prompt}". Output only the XML behavior tree without extra text.[/INST]
        RESPONSE: {fix_mistake}
        '''        


        if which_prompt == 1:
            prompt_template = xml_examples
        elif which_prompt == 2:
            prompt_template = zero_shot_learning
        elif which_prompt == 3:
            prompt_template = one_shot_learning
        elif which_prompt == 4:
            prompt_template = two_shot_learning


            
            

        print("###### prompt_template #######")
        print(prompt_template)

        
        try:
            response = self.llm(
                                prompt=prompt_template, 
                                max_tokens=self.context_length, 
                                temperature=self.temperature, 
                                top_p= self.top_p, 
                                repeat_penalty=self.repeat_penalty, 
                                top_k= self.top_k, 
                                echo=self.echo,
                                seed=self.seed                    
                                )  

            response = response["choices"][0]["text"]

        except ValueError as e:
            response = "Prompt length exceeds the maximum context length available for the model."

        
        
        
        print("###### response #######")
        print(response)

        
        behavior_tree = self.extract_behavior_tree(response)

        if behavior_tree == "Could not generate proper XML.":
            return self.error_count, behavior_tree
        elif behavior_tree == "try again":
            fix_mistake = f"""Here is your preivious respons: {response} it is incorrect.
            Ensure that it adheres to the correct structure, starting with the "<BehaviorTree>" root tag and ending with the "</BehaviorTree>" closing root tag. 
            USER COMMAND: generate behavior tree to "{prompt}. Output only the XML behavior tree without extra text.
            RESPONSE 2:"""
            return self.generate_behavior_tree(prompt, fix_mistake)
        else:
            return self.error_count, behavior_tree
        

    @staticmethod
    def call_behaviors():

        from main import SwarmAgent
        class_obj = SwarmAgent

        function_names_and_docstrings = {}
        for func_name in dir(class_obj):
            if callable(getattr(class_obj, func_name)) and not func_name.startswith("__")\
                and not func_name.startswith("update")\
                and not func_name.startswith("obstacle")\
                    and getattr(class_obj, func_name).__qualname__.startswith(class_obj.__name__ + "."):
                func = getattr(class_obj, func_name)
                if func.__doc__:
                    function_names_and_docstrings[func_name] = func.__doc__.strip()
                else:
                    function_names_and_docstrings[func_name] = "No docstring found."

        return function_names_and_docstrings
    
    def extract_behavior_tree(self, response):
        start_tag = "<BehaviorTree>"
        end_tag = "</BehaviorTree>"
        start_index = response.find(start_tag)
        end_index = response.find(end_tag)
        if start_index == -1 or end_index == -1:
            if self.error_count == 4:
                return 'Could not generate proper XML.'
            self.error_count += 1
            return "try again"
                
        behavior_tree_xml = response[start_index:end_index + len(end_tag)].strip()
        return behavior_tree_xml

def write_results(prompt, error_count, behavior_tree, model_name, filename):
    # Define the path for the CSV file
    file_path = os.path.join(os.getcwd(), filename)
    
    # Check if the file exists to decide if we need to write headers
    file_exists = os.path.isfile(file_path)
    
    # Field names for the CSV
    fieldnames = ['Prompt', 'Error Count', 'Behavior Tree', 'Model Name']
    
    # Open the file in append mode ('a') so we don't overwrite existing data
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write the header only if the file does not exist
        if not file_exists:
            writer.writeheader()
        
        # Data row to be written
        data_row = {
            'Prompt': prompt,
            'Error Count': error_count,
            'Behavior Tree': behavior_tree,
            'Model Name': model_name
        }
        
        # Write data row
        writer.writerow(data_row)



def main(model, prompt):

    fix_mistake=""
     
    error_count, behavior_tree = model.generate_behavior_tree(prompt=prompt, fix_mistake=fix_mistake)
    return error_count, behavior_tree





if __name__ == "__main__":

    models= [
        
            # ["TheBloke/Orca-2-7B-GGUF",
            # "/home/mohammed/Desktop/models/orca-2-7b.Q4_K_M.gguf"],

            # ["TheBloke/WestLake-7B-v2-GGUF",
            # "/home/mohammed/Desktop/models/westlake-7b-v2.Q4_K_M.gguf"],

            # ["TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            # "/home/mohammed/Desktop/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"],

            # ["TheBloke/Llama-2-7B-32K-Instruct-GGUF",
            # "/media/mohammed/AA5EEBF25EEBB4EB/llama-2-7b-32k-instruct.Q4_K_M.gguf"],

            # ["TheBloke/Llama-2-7b-Chat-GGUF",
            # "/media/mohammed/AA5EEBF25EEBB4EB/llama-2-7b-chat.Q4_K_M.gguf"],

            ["PawanKrd/phi-2.Q6_K.gguf",        
            "/home/mohammed/Desktop/models/phi-2.Q6_K.gguf"],

            # ["TheBloke/Llama-2-13B-GGUF",
            # "/home/mohammed/Desktop/models/llama-2-13b.Q4_K_M.gguf"],

            # ["TheBloke/Llama-2-13B-chat-GGUF",        
            # "/home/mohammed/Desktop/models/llama-2-13b-chat.Q4_K_M.gguf"],
            
            # ["PawanKrd/Meta-Llama-3-8B-Instruct-GGUF",        
            # "/home/mohammed/Desktop/models/llama-3-8b-instruct.Q4_K_M.gguf"],


            
            ]   


    scenarios = [
        "Find the target and change color to green, then return home and change color to white.",
        # "Find food and change color to green, then return to the nest, change color to white, and freeze movement.",
        # "Detect an obstacle, change color to green, try to find the goal, then change color to white.",
        # "Form a line and then change color to green.",
        # "change color to green, Form a line, and change color to white.",
        # "form vertical line", 
    ]


    # filenames = ["xml_examples_1.csv","zero_shot_learning_new_1.csv","one_shot_learning_new_1.csv","two_shot_learning_new_1.csv"]
    filenames = ["phi_2.csv"]

    for index in range(len(models)):
        
        which_prompt = 3

        config = {
        "model_file" : models[index][1],
        "context_length": (1024*4),
        "gpu_layers": 4,
        "n_threads":4,
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 150,
        "repeat_penalty": 1.2,
        "seed": 100
        }
        
        print("\n\n\n\n####", models[index][1]," ####\n\n\n\n")
        model_instance = BehaviorTreeGenerator(**config)  
        
        for filename in filenames:           

            for prompt in scenarios:            
                model_name = models[index][1].split("/")[-1].split(".")[0]
                error_count, behavior_tree = main(model_instance, prompt)
                write_results(prompt, error_count ,behavior_tree, model_name ,filename)
        
            which_prompt += 1