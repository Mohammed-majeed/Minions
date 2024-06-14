from ctransformers import AutoModelForCausalLM
import xml.etree.ElementTree as ET


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

        
        # BT_1 = """
        # <BehaviorTree>
        #     <Sequence>
        #         <Selector>
        #             <Sequence>
        #                 <Condition>target_detected</Condition>
        #                 <Sequence>
        #                     <Condition>path_clear</Condition>
        #                     <Action>target_reached</Action>
        #                 </Sequence>
        #                 <Sequence>
        #                     <Condition>path_clear</Condition>
        #                     <Action>avoid_obstacle</Action>
        #                 </Sequence>
        #             </Sequence>
        #             <Action>wander</Action>
        #         </Selector>
        #         <Selector>
        #             <Sequence>
        #                 <Condition>target_reached</Condition>
        #                 <Selector>
        #                     <Sequence>
        #                         <Condition>looking_for_nest</Condition>
        #                         <Action>return_to_nest</Action>
        #                     </Sequence>
        #                     <Sequence>
        #                         <Condition>path_clear</Condition>
        #                         <Action>avoid_obstacle</Action>
        #                     </Sequence>
        #                 </Selector>
        #             </Sequence>
        #             <Action>wander</Action>
        #         </Selector>
        #     </Sequence>
        # </BehaviorTree>
        # """

        # BT_2 = """
        # <BehaviorTree>
        #     <Selector>
        #         <Sequence>
        #             <Condition>obstacle_detected</Condition>
        #             <Action>avoid_obstacle</Action>
        #         </Sequence>
        #         <Action>form_line</Action>
        #     </Selector>
        # </BehaviorTree>
        # """

        # BT_3 = """
        # <BehaviorTree>
        #     <Action>change_color</Action>
        # </BehaviorTree>
        # """

        # read_examples = f"""
        # generate a behavior tree to find the target then back to the home
        # RESPONSE: {BT_1}

        # generate a behavior tree to change color
        # RESPONSE: {BT_3}
        # """


        BT_1 = """
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
        BT_2 = """
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
        

        tow_shut_learning = f'''<<SYS>>You are a helpful, respectful, and honest AI assistant. Your task is to generate well-structured XML code for behavior trees based on the provided instructions.<</SYS>>
        <s> [INST] 
        INSTRUCTIONS: Use only the following behaviors: {self.call_behaviors()} to construct behavior tree in XML format like the exampels to the following command.
        USER COMMAND: generate behavior tree to "Form a line, change color to green, and then freeze movement."[/INST]
        RESPONSE: {BT_1}
        </s>

        <<SYS>>You are a helpful, respectful, and honest AI assistant. Your task is to generate well-structured XML code for behavior trees based on the provided instructions.<</SYS>>
        <s> [INST]
        INSTRUCTIONS: Use only the following behaviors: {self.call_behaviors()} to construct behavior tree in XML format like the exampels to the following command.
        USER COMMAND: generate behavior tree to "Find food and change color to green, then return to the nest, change color to white."[/INST]
        RESPONSE: {BT_2}
        </s>
        '''



        prompt_template = f'''
        
        {tow_shut_learning}
        <<SYS>>You are a helpful, respectful, and honest AI assistant. Your task is to generate well-structured XML code for behavior trees based on the provided instructions.<</SYS>>
        <s> [INST]
        INSTRUCTIONS: Use only the following behaviors: {self.call_behaviors()} to construct behavior tree in XML format like the exampels to the following command.
        USER COMMAND: generate behavior tree to "{prompt}".[/INST]
        RESPONSE: {fix_mistake}
        '''

        print("###### prompt_template #######")
        print(prompt_template)

        response = self.llm(prompt_template)
        print("###### response #######")
        print(response)

        
        behavior_tree = self.extract_behavior_tree(prompt, response)
        return behavior_tree
        

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
        start_index = response.find(start_tag) + len(start_tag)
        end_index = response.find(end_tag)
        if start_index == -1 or end_index == -1:
            if self.error_count == 4:
                print('Could not generate proper XML.')
                exit()  
            self.error_count += 1
            fix_mistake = f"""you generated XML behavior tree structure is incorrect: {response}.
            Ensure that it adheres to the correct structure, starting with the "<BehaviorTree>" root tag and ending with the "</BehaviorTree>" closing root tag. User command:"{prompt}".
            Response 2:"""
            return self.generate_behavior_tree(prompt, fix_mistake)
        
        behavior_tree_xml = response[start_index:end_index].strip()
        return behavior_tree_xml

def save_behavior_tree_xml(data: str, file_path: str):
    if not data:
        print("No valid XML content to save.")
        return
    root = ET.fromstring(data)
    tree = ET.ElementTree(root)
    tree.write(file_path, encoding="utf-8", xml_declaration=True)        


def main(prompt, file_name):
    config = {

        # "model_id": "TheBloke/Orca-2-7B-GGUF",
        # "model_file":"/home/mohammed/Desktop/orca-2-7b.Q4_K_M.gguf",

        "model_id": "TheBloke/WestLake-7B-v2-GGUF",
        "model_file":"/home/mohammed/Desktop/models/westlake-7b-v2.Q4_K_M.gguf",

        # "model_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        # "model_file": "/home/mohammed/Desktop/mistral-7b-instruct-v0.2.Q4_K_M.gguf",

        # "model_id": "TheBloke/Llama-2-7B-32K-Instruct-GGUF",
        # "model_file":"/media/mohammed/AA5EEBF25EEBB4EB/llama-2-7b-32k-instruct.Q4_K_M.gguf",

        # "model_id": "TheBloke/Llama-2-7b-Chat-GGUF",
        # "model_file": "/media/mohammed/AA5EEBF25EEBB4EB/llama-2-7b-chat.Q4_K_M.gguf",

        # "model_id": "TheBloke/Llama-2-13B-GGUF",
        # "model_file": "/media/mohammed/AA5EEBF25EEBB4EB/llama-2-13b.Q4_K_M.gguf",

        # "model_id": "TheBloke/Llama-2-13B-chat-GGUF",        
        # "model_file": "/media/mohammed/AA5EEBF25EEBB4EB/llama-2-13b-chat.Q4_K_M.gguf",
        
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
    save_behavior_tree_xml(behavior_tree, file_name)






if __name__ == "__main__":
    prompt = "Find food and change color to green, then return to the nest, change color to white, and freeze movement."
    main(prompt, file_name="new_behaviors.xml")
