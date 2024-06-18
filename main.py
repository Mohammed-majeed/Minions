import pygame as pg
from pygame.math import Vector2
from vi import Agent, Simulation, Config, Window
from BT_parser import parse_behavior_tree
from LLAMA_2 import main
import math
import gc
import torch
import time
start_time = time.time()

class SwarmAgent(Agent):
    def __init__(self, images, simulation, pos, nest_pos, target_pos):
        super().__init__(images=images, simulation=simulation)
        self.root_node = parse_behavior_tree(xml_path)  
        self.pos = pos
        self.nest_pos = nest_pos
        self.target_pos = target_pos  
        self.target_detected_flag = False
        self.target_reached_flag = False
        self.obstacle_radius = 5
        self.state = "seeking"

    def update(self):
        self.root_node.run(self)


 
    def obstacle(self):
        """
        Check for obstacle intersections within a predefined radius.
        
        Returns: True if an obstacle is detected within the radius, False otherwise.
        """
        for intersection in self.obstacle_intersections(scale=self.obstacle_radius):
            return True
        return False

    def is_obstacle_detected(self):
        """
        Condition node: Determine if any obstacles are detected in the vicinity of the agent.
        Returns: True if an obstacle is detected, False otherwise.
        """
        if self.obstacle():
            return True
        else:
            return False

    def avoid_obstacle(self):
        """
        Action node: Execute an action to avoid detected obstacles.
        Returns: Always returns True, indicating the action was executed.
        """
        return True


    def is_target_detected(self):
        """
        Action node: Check if the target is within a detectable distance from the agent's position.
        Returns: True if the target is within 20 units of distance, False otherwise.
        """
        distance = math.dist(self.target_pos, self.pos)
        if distance <= 20:
            self.target_detected_flag = True        
        if self.target_detected_flag:
                    
            return True
        return False
        
    
    def is_target_reached(self):
        """
        Condition node: Check if the agent has reached the target.
        Returns: True if the target is within 15 units of distance, False otherwise.
        """
        distance = math.dist(self.target_pos, self.pos)
        if distance <= 15:
            self.target_reached_flag = True        
        if self.target_reached_flag:            
            return True
        return False
        

    def change_color_to_green(self):
        """
        Action node: Change the agent's color to green, usually indicating a successful operation or state.
        Returns: Always returns True, indicating the action was executed.
        """
        self.change_image(1)
        return True
    
    def change_color_to_white(self):
        """
        Action node: Change the agent's color to white, usually indicating a neutral or initial state.
        Returns: Always returns True, indicating the action was executed.
        """
        self.change_image(0)  
        return True
    

    def is_agent_in_nest(self):
        """
        Condition node: Determine if the agent is in the nest.
        Returns: True if the agent is in the nest, False otherwise.
        """
        distance = math.dist(self.nest_pos, self.pos)
        # elapsed_time = time.time() - start_time
        # if elapsed_time>2:
        if distance <= 17 and (self.target_reached_flag==True or self.target_detected_flag == True) :
            # self.state = "completed"
            # # self.freeze_movement()
            self.state = "seeking"
            self.target_detected_flag = False
            self.target_reached_flag = False
            return True
        return False


    def agent_movement_freeze(self):
        """
        Action node: Freeze the agent's movement, typically to indicate a stop in activity or end of tasks.
        Returns: Always returns True, indicating the action was executed.
        """
        self.freeze_movement()
        return True
    
    def continue_movement_agent(self):
        """
        Action node: Continue the agent's movement after it has been previously frozen.
        Returns: Always returns True, indicating the action was executed.
        """
        self.continue_movement()
        return True

    def wander(self):
        """
        Action node: Perform a wandering action where the agent moves randomly within the environment.
        Returns: Always returns True, indicating the action was executed.
        """
        super().change_position()
        return True

    def is_path_clear(self):
        """
        Condition node: Check if the path ahead of the agent is clear of obstacles.
        Returns: True if no obstacles are detected ahead, False if obstacles are present.
        """
        return not self.obstacle()
    
    def is_line_formed(self):
        """
        Condition node: Determine if the agent has formed a line with a reference point at the center of the window.
        Returns: True if the line is formed with the center, False otherwise.
        """
        center_x = self.config.window.width / 2
        direction = Vector2(center_x, self.pos.y) - self.pos
        if direction.length() > 0.5:
            return False        
        return True

    def form_line(self):
        """
        Action node: Direct the agent to form a line towards the center of the window. This function adjuststhe agent's position to align it with the center.
        Returns: Always returns True, indicating the action was executed.
        """
        center_x = self.config.window.width / 2
        direction = Vector2(center_x, self.pos.y) - self.pos
        if direction.length() > 0.5:
            direction.scale_to_length(self.config.movement_speed)
            self.pos += direction     
        return True
    
    def task_completed(self):
        """
        Action node: Signal that the agent has completed its designated task by freezing movement and updating state.
        Returns: Always returns True, indicating that the task completion action was executed.
        """
        self.state = "completed"
        return True
    

def draw_obstacle():
    x = 350
    y = 100
    simulation.spawn_obstacle("examples/images/rect_obst.png", x, y)
    simulation.spawn_obstacle("examples/images/rect_obst (1).png", y, x)

def draw_target(simulation):
    x = target_x
    y = target_y
    simulation.spawn_site("examples/images/rect.png", x, y)

def nest():
    x = nest_x
    y = nest_y
    simulation.spawn_site("examples/images/nest.png", x, y)

def load_images(image_paths):
    return [pg.image.load(path).convert_alpha() for path in image_paths]

if __name__ == '__main__':

    # gc.collect()

    # torch.cuda.empty_cache()

    xml_path = "behavior_tree_test.xml"
    # prompt = input('What behavior would you like to generate: ')
    # main(prompt=prompt, file_name=xml_path)

    config = Config(radius=50, visualise_chunks=True, window=Window.square(500))
    simulation = Simulation(config)

    # Nest position as a Vector2 object
    nest_x, nest_y = 450, 400  # Define the nest's position
    nest_pos = Vector2(nest_x, nest_y)

    # Define the target's position
    target_x, target_y = 200, 100
    target_pos = Vector2(target_x, target_y)

    # Ensure this path is correct and load images
    agent_images_paths = ["examples/images/white.png", "examples/images/green.png"]
    loaded_agent_images = load_images(agent_images_paths)  # Load images into Pygame surfaces

    # Initialize agents with loaded images and a starting position
    for _ in range(50):
        agent = SwarmAgent(images=loaded_agent_images, simulation=simulation, pos=Vector2(nest_x, nest_y),
                           nest_pos=nest_pos, target_pos=target_pos)
        simulation._agents.add(agent)
        simulation._all.add(agent)

    draw_obstacle()
    draw_target(simulation)
    nest()

    simulation.run()


