from dotenv import load_dotenv
import random
import json
import os
from openai import OpenAI
import pandas as pd
import anthropic




# Load environment variables
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# client = anthropic.Anthropic(
#   # defaults to os.environ.get("ANTHROPIC_API_KEY")
#   api_key="my_api_key",
# )


def encode_prompt(prompt_instructions):
        """Encode multiple prompt instructions into a single string."""
        prompt = open("./prompt.txt").read() + "\n"

        for idx, task_dict in enumerate(prompt_instructions):
            # USER_COMMAND, instruction, input, output = task_dict["USER COMMAND"], task_dict["INSTRUCTIONS"], task_dict["INPUT"], task_dict["OUTPUT"]
            instruction, USER_COMMAND, output =  task_dict["INSTRUCTIONS"], task_dict["USER COMMAND"], task_dict["OUTPUT"]
            # print(str(instruction)[1:-1])
            
            instruction = str(instruction)[1:-1]            
            instruction = instruction.replace("'Use only the following behaviors':","Use only the following behaviors")
            instruction = instruction.replace("'to construct behavior tree in XML format to the following command':","to construct behavior tree in XML format to the following command.")
            instruction = instruction.replace('"if the available behaviors do not fit with the user command say (Sorry, I can\'t do the task)"',"If the available behaviors do not fit with the user command say (Sorry, I can\'t do the task)")             

            prompt += f"\n###\n"
            prompt += f"{idx + 1}.// INSTRUCTIONS: {instruction}\n"
            prompt += f"{idx + 1}.// USER COMMAND: {USER_COMMAND}\n"
            prompt += f"{idx + 1}.// OUTPUT:\n{output}\n"
        # prompt += f"###\n"
        # prompt += f"{idx + 2}. USER COMMAND:"
        return prompt




for i in range(500):
    print("###########,",i,",############")

    # Open and read the JSON file
    with open("original seed task.json", 'r') as file:
        data = json.load(file)



    # seed_instruction_data = [{"INSTRUCTIONS": t["INSTRUCTIONS"],"USER COMMAND": t["USER COMMAND"], "INPUT": t["INSTANCES"][0]["INPUT"], "OUTPUT": t["INSTANCES"][0]["OUTPUT"]} for t in data]
    seed_instruction_data = [{"INSTRUCTIONS": t["INSTRUCTIONS"],"USER COMMAND": t["USER COMMAND"], "OUTPUT": t["OUTPUT"]} for t in data]


    # Determine batch size randomly between 1 and 4
    batch_size = random.randint(2, 3)
    # Select a random batch of the determined size
    batch_inputs = random.sample(seed_instruction_data, batch_size)

    prompt_tamplet = encode_prompt(batch_inputs)

    with open("prompt_22.txt", "w") as file:
        file.write(prompt_tamplet)




    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt_tamplet,
        temperature=1,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response)

    text = response.choices[0].text
    # print(text)

    with open("data_generated_10.txt", "a", encoding="utf-8") as file:
        file.write(text)
