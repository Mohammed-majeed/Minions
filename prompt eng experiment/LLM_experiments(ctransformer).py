from ctransformers import AutoModelForCausalLM
import xml.etree.ElementTree as ET
import csv
import os

class BehaviorTreeGenerator:
    def __init__(self, model_id, 
                 model_file, 
                 model_type, 
                 context_length,
                 gpu_layers,
                 temperature,
                 top_p,
                 top_k,
                 ):

        self.error_count = 0

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=model_id,
            model_file=model_file,
            model_type=model_type,
            context_length=context_length,
            gpu_layers=gpu_layers,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
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

        # read_examples = f"""
        # generate a behavior tree to find the target then back to the home
        # RESPONSE: {BT_1}

        # generate a behavior tree to change color
        # RESPONSE: {BT_2}
        # """

        # prompt_template_1 = f'''
        # SYSTEM: As a helpful, respectful, and honest AI assistant, your main task is to create well-structured XML behavior trees according to the instructions provided.
        # INSTRUCTIONS: You're given examples of behavior tree structures, like:\n{BT_1} \n or \n {BT_2}\n \nNote that while these examples can guide you, feel free to devise different structures as needed. Your objective is to use the following available behaviors: {self.call_behaviors()}, to craft a behavior tree in XML format. This tree should cater to simulated swarm robots based on the user's command below.
        # USER COMMAND: "{prompt}"
        # ASSISTANT RESPONSE: {fix_mistake}
        # '''
        # prompt_template_1 = f'''
        # SYSTEM: As a helpful, respectful, and honest AI assistant, your main task is to create well-structured XML behavior trees according to the instructions provided.
        # INSTRUCTIONS: You're given examples of behavior tree structures, like:
        # 1. Example 1:
        # {BT_1}
        # 2. Example 2:
        # {BT_2}
        # Note that while these examples can guide you, feel free to devise different structures as needed. Your objective is to use the following available behaviors: {self.call_behaviors()}, to craft a behavior tree in XML format. This tree should cater to simulated swarm robots based on the user's command below.
        # USER COMMAND: generate behavior tree to "{prompt}". Output only the XML behavior tree without extra text.[/INST]
        # RESPONSE: {fix_mistake}
        # '''

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
            prompt_template = two_shot_learning
        elif which_prompt == 3:
            prompt_template = one_shot_learning
        elif which_prompt == 4:
            prompt_template = zero_shot_learning
            
            

        print("###### prompt_template #######")
        print(prompt_template)

        response = self.llm(prompt_template)
        # print("###### response #######")
        # print(response)

        
        behavior_tree = self.extract_behavior_tree(prompt, response)

        if behavior_tree == "Could not generate proper XML.":
            return f"Iterations: {self.error_count}\n\nBehavior Tree XML:\n{behavior_tree}"
        elif behavior_tree == "try again":
            fix_mistake = f"""you generated XML behavior tree structure is incorrect: {response}.
            Ensure that it adheres to the correct structure, starting with the "<BehaviorTree>" root tag and ending with the "</BehaviorTree>" closing root tag. User command:"{prompt}".
            Response 2:"""
            return self.generate_behavior_tree(prompt, fix_mistake)
        else:
            return f"Iterations: {self.error_count}\n\nBehavior Tree XML:\n{behavior_tree}"

        # output = 
        
        # return output
        

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
    
    def extract_behavior_tree(self, prompt, response):
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

def save_behavior_tree_xml(data: str, file_path: str):
    if not data:
        print("No valid XML content to save.")
        return
    root = ET.fromstring(data)
    tree = ET.ElementTree(root)
    tree.write(file_path, encoding="utf-8", xml_declaration=True)        


def main(prompt, model_id, model_file):
    config = {
        "model_id": model_id,
        "model_file" : model_file,
        "model_type": "llama",
        "context_length": (1024*4),
        "gpu_layers": 4,
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 150

    }
    fix_mistake=""
    behavior_tree_generator = BehaviorTreeGenerator(**config)    
    behavior_tree = behavior_tree_generator.generate_behavior_tree(prompt=prompt, fix_mistake=fix_mistake)
    # save_behavior_tree_xml(behavior_tree, file_name)
    return behavior_tree



def write_results(prompt, result, model_name, filename):
    # Check if file exists and read existing data
    data = {}
    fieldnames = ['Prompt']
    if os.path.exists(filename):
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            fieldnames = reader.fieldnames
            for row in reader:
                data[row['Prompt']] = row

    # Prepare new fieldnames and data row
    if model_name not in fieldnames:
        fieldnames.append(model_name)
    data_row = data.get(prompt, {'Prompt': prompt})
    data_row[model_name] = result

    # Write to file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for row in data.values():
            writer.writerow(row)
        if prompt not in data:
            writer.writerow(data_row)





if __name__ == "__main__":

    models= [
        
            ["TheBloke/Orca-2-7B-GGUF",
              "/home/mohammed/Desktop/models/orca-2-7b.Q4_K_M.gguf"],

        #    ["TheBloke/WestLake-7B-v2-GGUF",
        #     "/home/mohammed/Desktop/models/westlake-7b-v2.Q4_K_M.gguf"],

        #     ["TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        #     "/home/mohammed/Desktop/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"],

        #     ["TheBloke/Llama-2-7B-32K-Instruct-GGUF",
        #     "/media/mohammed/AA5EEBF25EEBB4EB/llama-2-7b-32k-instruct.Q4_K_M.gguf"],

        #     ["TheBloke/Llama-2-7b-Chat-GGUF",
        #     "/media/mohammed/AA5EEBF25EEBB4EB/llama-2-7b-chat.Q4_K_M.gguf"],

        #     ["TheBloke/Llama-2-13B-GGUF",
        #     "/media/mohammed/AA5EEBF25EEBB4EB/llama-2-13b.Q4_K_M.gguf"],

        #     ["TheBloke/Llama-2-13B-chat-GGUF",        
        #     "/media/mohammed/AA5EEBF25EEBB4EB/llama-2-13b-chat.Q4_K_M.gguf"]
            
            ]   


    scenarios = [
        "Find the target and change color to green, then return home and change color to white.",
        # "Find food and change color to green, then return to the nest, change color to white, and freeze movement.",
        # "Detect an obstacle, change color to green, try to find the goal, then change color to white.",
        # "Form a line and then change color to green.",
        # "change color to green, Form a line, and change color to white.",
        # "form vertical line", 
    ]

    which_prompt = 1

    filenames = ["xml_examples.csv","two_shot_learning_new.csv","one_shot_learning_new.csv","zero_shot_learning_new.csv"]
    for index in range(len(filenames)):
        for prompt in scenarios:
            for model in models:
                model_name = model[1].split("/")[-1].split(".")[0]
                result = main(prompt, model[0],model[1])
                write_results(prompt, result, model_name ,filenames[index])
        
        which_prompt = index+1





