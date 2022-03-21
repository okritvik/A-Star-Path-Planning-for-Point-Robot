# A-Star.py
"""
Authors: Kumara Ritvik Oruganti (okritvik@umd.edu), 2022
         Adarsh Malapaka (amalapak@umd.edu), 2022
Brief: Computes and visualizes an optimal path between the start and goal posiitons in a 5-connected (-60,-30, 0, 30, 60) degrees
       obstacle map for a point mobile robot with non-zero radius and clearance and a defined step size using A* Algorithm. 
Course: ENPM662 - Planning for Autonomous Robotics [Project-03, Phase 01]
        University of Maryland, College Park (MD)
Date: 21st March, 2022
"""
# Importing the required libraries
import time
import cv2
import numpy as np
import heapq as hq


def take_robot_inputs():
    """
    Gets the robot radius and clearance inputs from the user.
                
    Parameters:
        None
    Returns:
        clearance: int
                Robot's clearance
        robot_radius: int 
                Robot's radius
    """
    clearance = 0
    robot_radius = 0

    while True:
        clearance = input("Enter the robot's clearance: ")
        if(int(clearance)<0):
            print("Enter a valid Robot Clearance!")
        else:
            break

    while True:
        robot_radius = input("Enter the robot's radius: ")
        if(int(robot_radius)<0):
            print("Enter a valid Robot Radius!")
        else:
            break
    
    return int(clearance), int(robot_radius)


def take_map_inputs(canvas):
    """
    Gets the initial node, final node coordinates, heading angles and step-size from the user.
                
    Parameters:
        canvas: NumPy array
                Map matrix
    Returns:
        initial_state: List
                List to hold the initial node coordinates and heading angle
        final_state: List 
                List to hold the final node coordinates and heading angle
        
        step: int 
                Robot's step size
    """
    initial_state = []
    final_state = []
    initial_angle = 0
    final_angle = 0
    step = 0

    while True:
        while True:
            state = input("Enter the X Coordinate of the Start Node: ")
            if(int(state)<0 or int(state)>canvas.shape[1]-1):
                print("Enter a valid X Coordinate!")
                continue
            else:
                initial_state.append(int(state))
                break
        while True:
            state = input("Enter the Y Coordinate of the Start Node: ")
            if(int(state)<0 or int(state)>canvas.shape[0]-1):
                print("Enter a valid Y Coordinate!")
                continue
            else:
                initial_state.append(int(state))
                break
        
        if(canvas[canvas.shape[0]-1 - initial_state[1]][initial_state[0]][0]==255):
            print("*** The entered start node is in the Obstacle Space! ***")
            initial_state.clear()
        else:
            break

    while True:
        while True:
            state = input("Enter the X Coordinate of the Goal Node: ")
            if(int(state)<0 or int(state)>canvas.shape[1]-1):
                print("Enter a valid X Coordinate!")
                continue
            else:
                final_state.append(int(state))
                break
        while True:
            state = input("Enter the Y Coordinate of the Goal Node: ")
            if(int(state)<0 or int(state)>canvas.shape[0]-1):
                print("Enter a valid Y Coordinate!")
                continue
            else:
                final_state.append(int(state))
            break

        if(canvas[canvas.shape[0]-1 - final_state[1]][final_state[0]][0]==255):
            print("*** The entered goal node is in the obstacle space! ***")
            final_state.clear()
        else:
            break
    while True:
        initial_angle = input("Enter the Initial Head Angle (+- multiple of 30 deg): ")
        
        # if(int(initial_angle)<0 or int(initial_angle)>359 or (int(initial_angle)%30 != 0)):
        if((int(initial_angle)%30 != 0)):
            print("Enter a valid Headway Angle!")
        else:
            if int(initial_angle) < 0:
                initial_angle = 360 + int(initial_angle)
            initial_state.append(int(initial_angle))
            break
            

    while True:
        final_angle = input("Enter the Final Head Angle (+- multiple of 30 deg): ")
        
        # if(int(initial_angle)<0 or int(initial_angle)>359 or (int(initial_angle)%30 != 0)):
        if((int(final_angle)%30 != 0)):
            print("Enter a valid Headway Angle!")
        else:
            if int(final_angle) < 0:
                final_angle = 360 + int(final_angle)
            final_state.append(int(final_angle))
            break

    while True:
        step = input("Enter the step size between 1 to 10: ")
        if(int(step)<1 and int(step)>10):
            print("Enter a valid step size!")
        else:
            break
    
    return initial_state,final_state,int(step)


def draw_obstacles(canvas, offset=15):
    """
    Draws the obstacles and walls in the map incorporating the robot's offset.
                
    Parameters:
        canvas: NumPy array
                Map matrix
        offset: int
                Offset is robot radius + clearance 
    Returns:
        canvas: NumPy array
                Map matrix with drawn obstacles
    """
    height, width, __ = canvas.shape

    for i in range(width):
        for j in range(height):
            if(i<=offset) or (i>=(400-offset)) or (j<=offset) or (j>=(250-offset)):
                canvas[j][i] = [255,0,0]

            # Drawing scaled obstacles with robot clearance and radius
            if ((i-300)**2+(j-65)**2-((40+offset)**2))<=0:
                canvas[j][i] = [255,0,0]
            
            if (j+(0.57*i)-(224.285-offset*1.151))>=0 and (j-(0.57*i)+(4.285+offset*1.151))>=0 and (i-(235+offset))<=0 and (j+(0.57*i)-(304.285+offset*1.151))<=0 and (j-(0.57*i)-(75.714+offset*1.151))<=0 and (i-(165-offset))>=0:
                canvas[j][i] = [255,0,0]

            if ((j+(0.316*i)-(76.392-offset*1.048)>=0) and (j+(0.857*i)-(138.571+offset*1.317)<=0) and (j-(0.114*i)-60.909)<=0) or ((j-(3.2*i)+(186+offset*3.352)>=0) and (j-(1.232*i)-(20.652+offset*1.586))<=0 and (j-(0.114*i)-60.909)>=0):
                canvas[j][i] = [255,0,0]

            # Drawing actual obstacles without robot clearance and radius
            if ((i-300)**2+(j-65)**2-((40)**2))<=0:
                canvas[j][i] = [255,255,0]
            
            if (j+(0.57*i)-(224.285))>=0 and (j-(0.57*i)+(4.285))>=0 and (i-(235))<=0 and (j+(0.57*i)-(304.285))<=0 and (j-(0.57*i)-(75.714))<=0 and (i-(165))>=0:
                canvas[j][i] = [255,255,0]

            if ((j+(0.316*i)-(76.392)>=0) and (j+(0.857*i)-(138.571)<=0) and (j-(0.114*i)-60.909)<=0) or ((j-(3.2*i)+(186)>=0) and (j-(1.232*i)-(20.652))<=0 and (j-(0.114*i)-60.909)>=0):
                canvas[j][i] = [255,255,0]

    return canvas


def threshold(num):
    """
    Rounds the given number to the nearest 0.5 value.
    For ex: 4.61 is rounded to 4.5 whereas 4.8 is rounded to 5.0.
                
    Parameters:
        num: float
                Number to be rounded
    Returns:
        num: float
                Number rounded to nearest 0.5
    """
    return round(num*2)/2


def check_goal(node, final):
    """
    Checks if the given current node is within the goal node's threshold distance of 1.5.
                
    Parameters:
        node: List
                Current node
        final: List
                Goal node 
    Returns:
        flag: bool
                True if the present node is the goal node, False otherwise
    """
    if(np.sqrt(np.power(node[0]-final[0],2)+np.power(node[1]-final[1],2))<1.5) and (node[2]==final[2]):
        return True
    else:
        return False


def cost_to_goal(node, final):
    """
    Computes the Cost To Goal between present and goal nodes using a Euclidean distance heuristic.
                
    Parameters:
        node: List
                Current node
        final: List
                Goal node 
    Returns:
        flag: bool
                True if the present node is the goal node, False otherwise
    """
    return np.sqrt(np.power(node[0]-final[0],2)+np.power(node[1]-final[1],2))


def check_obstacle(next_width, next_height, canvas):    
    """
    Checks if the generated/next node is in the obstacle region.
              
    Parameters:
        next_width: float
                Width of the next node from the present node
        next_height: float
                Height of the next node from the present node
        
        canvas: NumPy array
                Map matrix with drawn obstacles 
    Returns:
        flag: bool
                True if the next node is NOT in the obstacle region, False otherwise
    """
    if canvas[int(round(next_height))][int(round(next_width))][0]==255:
        return False
    else:
        return True


def action_zero(node, canvas, visited, step):    # Local angles
    """
    Moves the robot at 0 degree angle (wrt robot's frame) by the step amount. 
              
    Parameters:
        node: List
                List of node's x, y and theta parameters
        
        canvas: NumPy array
                Map matrix with drawn obstacles 
        visited: NumPy array
                Visited matrix of size 500x800x12 to keep track of duplicate nodes  
        step: int
               Step size of the robot 
    Returns:
        Next Node flag: bool
                True if the child node can be generated, False otherwise
        next_node: List
                Child node generated after performing the action
        Duplicate Node flag: bool
                True if generated next node is already visited, False otherwise
    """
    next_node = node.copy()
    next_angle = next_node[2] + 0    # Angle in Cartesian System

    if next_angle < 0:
        next_angle += 360 
    next_angle %= 360
    next_width = threshold(next_node[0] + step*np.cos(np.deg2rad(next_angle)))
    next_height = threshold(next_node[1] + step*np.sin(np.deg2rad(next_angle)))

    if (round(next_height)>0 and round(next_height)<canvas.shape[0]) and (round(next_width)>0 and round(next_width)<canvas.shape[1]) and (check_obstacle(next_width,next_height,canvas)) :
        next_node[0] = next_width
        next_node[1] = next_height
        next_node[2] = next_angle

        if(visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] == 1):
            return True, next_node, True
        else:
            visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] = 1
            return True, next_node, False
    else:
        return False, next_node, False
    

def action_minus_thirty(node, canvas, visited, step):    # Local angles
    """
    Moves the robot at -30 degree angle (wrt robot's frame) by the step amount. 
              
    Parameters:
        node: List
                List of node's x, y and theta parameters
        
        canvas: NumPy array
                Map matrix with drawn obstacles 
        visited: NumPy array
                Visited matrix of size 500x800x12 to keep track of duplicate nodes  
        step: int
               Step size of the robot 
    Returns:
        Next Node flag: bool
                True if the child node can be generated, False otherwise
        next_node: List
                Child node generated after performing the action
        Duplicate Node flag: bool
                True if generated next node is already visited, False otherwise
    """
    next_node = node.copy()
    next_angle = next_node[2] + 30    # Angle in Cartesian System
    
    if next_angle < 0:
        next_angle += 360 
    next_angle %= 360
    next_width = threshold(next_node[0] + step*np.cos(np.deg2rad(next_angle)))
    next_height = threshold(next_node[1] + step*np.sin(np.deg2rad(next_angle)))

    if (round(next_height)>0 and round(next_height)<canvas.shape[0]) and (round(next_width)>0 and round(next_width)<canvas.shape[1]) and (check_obstacle(next_width,next_height,canvas)) :
        next_node[0] = next_width
        next_node[1] = next_height
        next_node[2] = next_angle
        
        if(visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] == 1):
            return True, next_node, True
        else:
            visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] = 1
            return True, next_node, False
    else:
        return False, next_node, False


def action_minus_sixty(node, canvas, visited, step):    # Local angles
    """
    Moves the robot at -60 degree angle (wrt robot's frame) by the step amount. 
              
    Parameters:
        node: List
                List of node's x, y and theta parameters
        
        canvas: NumPy array
                Map matrix with drawn obstacles 
        visited: NumPy array
                Visited matrix of size 500x800x12 to keep track of duplicate nodes  
        step: int
               Step size of the robot 
    Returns:
        Next Node flag: bool
                True if the child node can be generated, False otherwise
        next_node: List
                Child node generated after performing the action
        Duplicate Node flag: bool
                True if generated next node is already visited, False otherwise
    """
    next_node = node.copy()
    next_angle = next_node[2] + 60    # Angle in Cartesian System
    
    if next_angle < 0:
        next_angle += 360
    
    next_angle %= 360 
    next_width = threshold(next_node[0] + step*np.cos(np.deg2rad(next_angle)))
    next_height = threshold(next_node[1] + step*np.sin(np.deg2rad(next_angle)))

    if (round(next_height)>0 and round(next_height)<canvas.shape[0]) and (round(next_width)>0 and round(next_width)<canvas.shape[1]) and (check_obstacle(next_width,next_height,canvas)) :
        next_node[0] = next_width
        next_node[1] = next_height
        next_node[2] = next_angle

        if(visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] == 1):
            return True, next_node,True
        else:
            visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] = 1
            return True, next_node, False
    else:
        return False, next_node, False


def action_plus_thirty(node, canvas, visited, step):    # Local angles
    """
    Moves the robot at +30 degree angle (wrt robot's frame) by the step amount. 
              
    Parameters:
        node: List
                List of node's x, y and theta parameters
        
        canvas: NumPy array
                Map matrix with drawn obstacles 
        visited: NumPy array
                Visited matrix of size 500x800x12 to keep track of duplicate nodes  
        step: int
               Step size of the robot 
    Returns:
        Next Node flag: bool
                True if the child node can be generated, False otherwise
        next_node: List
                Child node generated after performing the action
        Duplicate Node flag: bool
                True if generated next node is already visited, False otherwise
    """
    next_node = node.copy()
    next_angle = next_node[2] - 30    # Angle in Cartesian System

    if next_angle < 0:
        next_angle += 360 
    next_angle %= 360
    next_width = threshold(next_node[0] + step*np.cos(np.deg2rad(next_angle)))
    next_height = threshold(next_node[1] + step*np.sin(np.deg2rad(next_angle)))

    if (round(next_height)>0 and round(next_height)<canvas.shape[0]) and (round(next_width)>0 and round(next_width)<canvas.shape[1]) and (check_obstacle(next_width,next_height,canvas)) :
        next_node[0] = next_width
        next_node[1] = next_height
        next_node[2] = next_angle

        if(visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] == 1):
            return True, next_node, True
        else:
            visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] = 1
            return True, next_node, False
    else:
        return False, next_node, False


def action_plus_sixty(node, canvas, visited, step):    # Local angles
    """
    Moves the robot at +60 degree angle (wrt robot's frame) by the step amount. 
              
    Parameters:
        node: List
                List of node's x, y and theta parameters
        
        canvas: NumPy array
                Map matrix with drawn obstacles 
        visited: NumPy array
                Visited matrix of size 500x800x12 to keep track of duplicate nodes  
        step: int
               Step size of the robot 
    Returns:
        Next Node flag: bool
                True if the child node can be generated, False otherwise
        next_node: List
                Child node generated after performing the action
        Duplicate Node flag: bool
                True if generated next node is already visited, False otherwise
    """
    next_node = node.copy()
    next_angle = next_node[2] - 60    # Angle in Cartesian System

    if next_angle < 0:
        next_angle += 360
    next_angle %= 360
    next_width = threshold(next_node[0] + step*np.cos(np.deg2rad(next_angle)))
    next_height = threshold(next_node[1] + step*np.sin(np.deg2rad(next_angle)))

    if (round(next_height)>0 and round(next_height)<canvas.shape[0]) and (round(next_width)>0 and round(next_width)<canvas.shape[1]) and (check_obstacle(next_width,next_height,canvas)) :
        next_node[0] = next_width
        next_node[1] = next_height
        next_node[2] = next_angle

        if(visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] == 1):
            return True, next_node,True
        else:
            visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] = 1
            return True, next_node,False
    else:
        return False, next_node,False


def astar(initial_state, final_state, canvas, step):
    """
    Implements the A* algorithm to find the path between the user-given start node and goal node.  
    It is robust enough to raise a 'no solution' prompt for goal/start states in the obstacle space.
    The open list is a heap queue which uses the Total Cost as the key to sort the heap.
    The closed list is a dictionary with key as the current node and value as the parent node.
    
    Parameters:
        initial_state: List
                List of start node's x, y and theta parameters
        
        final_state: List
                List of goal node's x, y and theta parameters 
        canvas: NumPy array
                Map matrix with drawn obstacles  
        step: int
               Step size of the robot 
    Returns:
            None
    """
    open_list = []    # Format: {(TotalCost): CostToCome, CostToGo, PresentNode, ParentNode}
    closed_list = {}    # Format: {(PresentNode): ParentNode}
    back_track_flag = False
    
    visited_nodes = np.zeros((500,800,12))
    
    hq.heapify(open_list)
    present_c2c = 0
    present_c2g = cost_to_goal(initial_state,final_state)
    total_cost = present_c2c + present_c2g
    hq.heappush(open_list,[total_cost,present_c2c,present_c2g,initial_state,initial_state])
    
    while len(open_list)!=0:
        node = hq.heappop(open_list)
        # print("\nPopped node: ",node)
        closed_list[tuple(node[4])] = node[3]
        if(check_goal(node[4],final_state)):
            print("\nGoal Reached!")
            back_track_flag = True
            back_track(initial_state,node[4],closed_list,canvas)
            break

        present_c2c = node[1]
        present_c2g = node[2]
        total_cost = node[0]

        # Move +60 degrees
        flag, n_state, dup = action_plus_sixty(node[4],canvas,visited_nodes,step)    # flag is True if valid move
        if(flag):
            if tuple(n_state) not in closed_list:
                if(dup):
                    for i in range(len(open_list)):
                        if tuple(open_list[i][4]) == tuple(n_state):
                            cost = present_c2c+step+cost_to_goal(n_state,final_state)
                            if(cost<open_list[i][0]):    # Updating the cost and parent info of the node
                                open_list[i][1] = present_c2c+step
                                open_list[i][0] = cost
                                open_list[i][3] = node[4]
                                hq.heapify(open_list)
                            break
                else:
                    hq.heappush(open_list,[present_c2c+step+cost_to_goal(n_state,final_state),present_c2c+step,cost_to_goal(n_state,final_state),node[4],n_state])
                    hq.heapify(open_list)

        # Move +30 degrees
        flag, n_state, dup = action_plus_thirty(node[4],canvas,visited_nodes,step)    # flag is True if valid move
        if(flag):
            if tuple(n_state) not in closed_list:
                if(dup):
                    for i in range(len(open_list)):
                        if tuple(open_list[i][4]) == tuple(n_state):
                            cost = present_c2c+step+cost_to_goal(n_state,final_state)
                            if(cost<open_list[i][0]):    # Updating the cost and parent info of the node
                                open_list[i][1] = present_c2c+step
                                open_list[i][0] = cost
                                open_list[i][3] = node[4]
                                hq.heapify(open_list)
                            break
                else:
                    hq.heappush(open_list,[present_c2c+step+cost_to_goal(n_state,final_state),present_c2c+step,cost_to_goal(n_state,final_state),node[4],n_state])
                    hq.heapify(open_list)
                
        # Move 0 degrees
        flag, n_state, dup = action_zero(node[4],canvas,visited_nodes,step)    # flag is True if valid move
        if(flag):
            if tuple(n_state) not in closed_list:
                if(dup):
                    for i in range(len(open_list)):
                        if tuple(open_list[i][4]) == tuple(n_state):
                            cost = present_c2c+step+cost_to_goal(n_state,final_state)
                            if(cost<open_list[i][0]):    # Updating the cost and parent info of the node
                                open_list[i][1] = present_c2c+step
                                open_list[i][0] = cost
                                open_list[i][3] = node[4]
                                hq.heapify(open_list)
                            break
                else:
                    hq.heappush(open_list,[present_c2c+step+cost_to_goal(n_state,final_state),present_c2c+step,cost_to_goal(n_state,final_state),node[4],n_state])
                    hq.heapify(open_list)

        # Move -30 degrees
        flag, n_state, dup = action_minus_thirty(node[4],canvas,visited_nodes,step)    # flag is True if valid move
        if(flag):
            if tuple(n_state) not in closed_list:
                if(dup):
                    for i in range(len(open_list)):
                        if tuple(open_list[i][4]) == tuple(n_state):
                            cost = present_c2c+step+cost_to_goal(n_state,final_state)
                            if(cost<open_list[i][0]):    # Updating the cost and parent info of the node
                                open_list[i][1] = present_c2c+step
                                open_list[i][0] = cost
                                open_list[i][3] = node[4]
                                hq.heapify(open_list)
                            break
                else:
                    hq.heappush(open_list,[present_c2c+step+cost_to_goal(n_state,final_state),present_c2c+step,cost_to_goal(n_state,final_state),node[4],n_state])
                    hq.heapify(open_list)

        # Move -60 degrees
        flag,n_state,dup = action_minus_sixty(node[4],canvas,visited_nodes,step)    # flag is True if valid move
        if(flag):
            if tuple(n_state) not in closed_list:
                if(dup):
                    for i in range(len(open_list)):
                        if tuple(open_list[i][4]) == tuple(n_state):
                            cost = present_c2c+step+cost_to_goal(n_state,final_state)
                            if(cost<open_list[i][0]):    # Updating the cost and parent info of the node
                                open_list[i][1] = present_c2c+step
                                open_list[i][0] = cost
                                open_list[i][3] = node[4]
                                hq.heapify(open_list)
                            break
                else:
                    hq.heappush(open_list,[present_c2c+step+cost_to_goal(n_state,final_state),present_c2c+step,cost_to_goal(n_state,final_state),node[4],n_state])
                    hq.heapify(open_list)

    if not back_track_flag:    
        print("\nNo Solution Found!")
        print("Total Number of Nodes Explored: ",len(closed_list))


def back_track(initial_state, final_state, closed_list, canvas):
    """
    Implements backtracking to the start node after reaching the goal node.
    This function is also used for visualization of explored nodes and computed path using OpenCV.
    A stack is used to store the intermediate nodes while transversing from the goal node to start node.
    
    Parameters:
        initial_state: List
                List of start node's x, y and theta parameters
        
        final_state: List
                List of goal node's x, y and theta parameters 
        closed_list: Dictionary
                Dictionary containing explored nodes and corresponding parents  
        canvas: NumPy array
                Map matrix with drawn obstacles 
    Returns:
            None
    """

    fourcc = cv2.VideoWriter_fourcc(*'XVID')    # Creating video writer to generate a video.
    out = cv2.VideoWriter('A-Star-amalapak-okritvik_testCase01.avi',fourcc,500,(canvas.shape[1],canvas.shape[0]))
    
    print("Total Number of Nodes Explored = ",len(closed_list)) 
    
    keys = closed_list.keys()    # Returns all the nodes that are explored
    path_stack = []    # Stack to store the path from start to goal
    
    # Visualizing the explored nodes
    keys = list(keys)
    for key in keys:
        p_node = closed_list[tuple(key)]
        cv2.circle(canvas,(int(key[0]),int(key[1])),2,(0,0,255),-1)
        cv2.circle(canvas,(int(p_node[0]),int(p_node[1])),2,(0,0,255),-1)
        canvas = cv2.arrowedLine(canvas, (int(p_node[0]),int(p_node[1])), (int(key[0]),int(key[1])), (0,255,0), 1, tipLength = 0.2)
        cv2.imshow("A* Exploration and Optimal Path Visualization",canvas)
        cv2.waitKey(1)
        out.write(canvas)

    parent_node = closed_list[tuple(final_state)]
    path_stack.append(final_state)    # Appending the final state because of the loop starting condition
    
    while(parent_node != initial_state):
        path_stack.append(parent_node)
        parent_node = closed_list[tuple(parent_node)]
    
    path_stack.append(initial_state)    # Appending the initial state because of the loop breaking condition
    print("\nOptimal Path: ")
    start_node = path_stack.pop()
    print(start_node)

    # Visualizing the optimal path
    while(len(path_stack) > 0):
        path_node = path_stack.pop()
        cv2.line(canvas,(int(start_node[0]),int(start_node[1])),(int(path_node[0]),int(path_node[1])),(255,0,196),5)
        print(path_node)
        start_node = path_node.copy()
        out.write(canvas)
    
    out.release()

if __name__ == '__main__':
    
    canvas = np.ones((250,400,3), dtype="uint8")    # Creating a blank canvas/map
    clearance, robot_radius = take_robot_inputs()
    canvas = draw_obstacles(canvas,offset = (clearance + robot_radius))
    initial_state, final_state, step = take_map_inputs(canvas) #Take the start and goal node from the user
    
    # Uncomment the below 3 lines to view the obstacle space. Press Any Key to close the image window
    # cv2.imshow("Canvas",canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Changing the input Cartesian Coordinates of the Map to Image Coordinates:
    initial_state[1] = canvas.shape[0]-1 - initial_state[1]
    final_state[1] = canvas.shape[0]-1 - final_state[1]
    # print(initial_state, final_state)

    # Converting the angles with respect to the image coordinates
    if initial_state[2] != 0:
        initial_state[2] = 360 - initial_state[2]
    if final_state[2] != 0:
        final_state[2] = 360 - final_state[2]

    print("\nStart Node (image coords): ", initial_state)
    print("Goal Node (image coords): ", final_state)

    start_time = time.time()

    cv2.circle(canvas,(int(initial_state[0]),int(initial_state[1])),2,(0,0,255),-1)
    cv2.circle(canvas,(int(final_state[0]),int(final_state[1])),2,(0,0,255),-1)
    
    astar(initial_state,final_state,canvas,step)    # Compute the optimal path using A* Algorithm
    
    end_time = time.time()    # Time taken for the algorithm to find the optimal path
    print("\nCode Execution Time (sec): ", end_time-start_time)    # Computes & prints the total execution time

    cv2.imshow("A* Exploration and Optimal Path Visualization", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
         
