"""
@author Kumara Ritvik Oruganti
@brief This code is written as part of the Project 2 of ENPM 661 to find a path for the point robot using Dijkstra Algorithm
There will be 8 action sets, Up, Down, Left, Right and Four Diagonals.
Cost for up, down, left, right actions is 1 and for diagonal actions, the cost is 1.4.

A geometrical obstacle map is given. For the given obstacle map, mathematical equations are developed and with 5mm clearance, the obstacle map is generated using the half planes.
The path is found from a given start node to goal node and visualized using OpenCV.
"""

#Imports
import copy
from turtle import width
import numpy as np
import cv2
import heapq as hq
import time
import matplotlib.pyplot as plt

def take_inputs(canvas):
    """
    @brief: This function takes the initial node state and final node state to solve the puzzle.
    :param canvas: canvas image  
    Prompts the user to input again if the nodes are not positive and out of bounds
    :return: Initial state and final state of the puzzle to be solved
    """
    initial_state = []
    final_state = []
    initial_angle = 0
    final_angle = 0
    clearance = 0
    robot_radius = 0
    step_size = 0
    while True:
        while True:
            state = input("Enter the X Coordinate of Start Node: ")
            if(int(state)<0 or int(state)>canvas.shape[1]-1):
                print("Enter Valid X Coordinate")
                continue
            else:
                initial_state.append(int(state))
                break
        while True:
            state = input("Enter the Y Coordinate of Start Node: ")
            if(int(state)<0 or int(state)>canvas.shape[0]-1):
                print("Enter Valid Y Coordinate")
                continue
            else:
                initial_state.append(int(state))
                break
        # print(canvas[canvas.shape[0]-1 - initial_state[1]][initial_state[0]])
        if(canvas[canvas.shape[0]-1 - initial_state[1]][initial_state[0]][0]==255):
            print("The entered start node is in the obstacle space")
            initial_state.clear()
        else:
            break
    while True:
        while True:
            state = input("Enter the X Coordinate of Goal Node: ")
            if(int(state)<0 or int(state)>canvas.shape[1]-1):
                print("Enter Valid X Coordinate")
                continue
            else:
                final_state.append(int(state))
                break
        while True:
            state = input("Enter the Y Coordinate of Goal Node: ")
            if(int(state)<0 or int(state)>canvas.shape[0]-1):
                print("Enter Valid Y Coordinate")
                continue
            else:
                final_state.append(int(state))
            break
        # print(canvas[canvas.shape[0]-1 - final_state[1]][final_state[0]])
        if(canvas[canvas.shape[0]-1 - final_state[1]][final_state[0]][0]==255):
            print("The entered goal node is in the obstacle space")
            final_state.clear()
        else:
            break
    while True:
        initial_angle = input("Enter the Initial Head Angle (0 to 360 degrees (multiple of 30 degrees)): ")
        if(int(initial_angle)<0 or int(initial_angle)>359):
            print("Enter Valid Headway Angle")
        else:
            initial_state.append(int(initial_angle))
            break
    while True:
        final_angle = input("Enter the Final Head Angle (0 to 360 degrees (multiple of 30 degrees)): ")
        if(int(final_angle)<0 or int(final_angle)>359):
            print("Enter Valid Headway Angle")
        else:
            final_state.append(int(final_angle))
            break
    while True:
        clearance = input("Enter the clearance ")
        if(int(clearance)<0):
            print("Enter Valid Clearance")
        else:
            break
    while True:
        robot_radius = input("Enter the robot radius ")
        if(int(robot_radius)<0):
            print("Enter Valid robot_radius")
        else:
            final_state.append(int(robot_radius))
            break
    while True:
        step = input("Enter the step size from 1 to 10 ")
        if(int(step)<1 and int(step)>10):
            print("Enter Valid step size")
        else:
            break
    return initial_state,final_state,int(step)

def draw_obstacles(canvas):
    """
    @brief: This function goes through each node in the canvas image and checks for the
    obstacle space using the half plane equations. 
    If the node is in obstacle space, the color is changed to blue.
    :param canvas: Canvas Image
    """
    # Uncomment to use the cv2 functions to create the obstacle space
    # cv2.circle(canvas, (300,65),45,(255,0,0),-1)
    # cv2.fillPoly(canvas, pts = [np.array([[115,40],[36,65],[105,150],[80,70]])], color=(255,0,0)) #Arrow
    # cv2.fillPoly(canvas, pts = [np.array([[200,110],[235,130],[235,170],[200,190],[165,170],[165,130]])], color=(255,0,0)) #Hexagon
    
    height,width,_ = canvas.shape
    # print(shape)
    for i in range(width):
        for j in range(height):
            if(i-5<=0) or (i-395>=0) or (j-5<=0) or (j-245>=0):
                canvas[j][i] = [255,0,0]

            if ((i-300)**2+(j-65)**2-(45**2))<=0:
                canvas[j][i] = [255,0,0]
            
            if (j+(0.57*i)-218.53)>=0 and (j-(0.57*i)+10.04)>=0 and (i-240)<=0 and (j+(0.57*i)-310.04)<=0 and (j-(0.57*i)-81.465)<=0 and (i-160)>=0:
                canvas[j][i] = [255,0,0]

            if ((j+(0.316*i)-71.1483)>=0 and (j+(0.857*i)-145.156)<=0 and (j-(0.114*i)-60.909)<=0) or ((j-(1.23*i)-28.576)<=0 and (j-(3.2*i)+202.763)>=0 and (j-(0.114*i)-60.909)>=0):
                canvas[j][i] = [255,0,0]
    return canvas

#Change the data structure to add total cost and cost to goal.
#Generate action Sets as given in PPT
def threshold(num):
    return round(num*2)/2

def action_zero(node,canvas,visited,step): # Local angles
    next_node = node.copy()
    
    next_angle = next_node[2] + 0
    if next_angle < 0:
        next_angle += 360 
    next_angle %= 360
    next_width = threshold(next_node[0] + step*np.cos(np.deg2rad(next_angle)))
    next_height = threshold(next_node[1] + step*np.sin(np.deg2rad(next_angle)))
    # print("Next Angle ",next_angle)
    # print("Width, Height: ",next_width,next_height)

    if (next_height>0 and next_height<canvas.shape[0]) and (next_width>0 and next_width<canvas.shape[1]) and (canvas[int(round(next_height))][int(round(next_width))][0]<255) :
        next_node[0] = next_width
        next_node[1] = next_height
        next_node[2] = next_angle
        visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] = 1
        return True, next_node
    else:
        return False, next_node
    

def action_minus_thirty(node,canvas,visited,step): # Local angles
    next_node = node.copy()
    
    next_angle = next_node[2] - 30
    if next_angle < 0:
        next_angle += 360 
    next_angle %= 360
    next_width = threshold(next_node[0] + step*np.cos(np.deg2rad(next_angle)))
    next_height = threshold(next_node[1] + step*np.sin(np.deg2rad(next_angle)))
    # print("Next Angle ",next_angle)
    # print("Width, Height: ",next_width,next_height)

    if (next_height>0 and next_height<canvas.shape[0]) and (next_width>0 and next_width<canvas.shape[1]) and (canvas[int(round(next_height))][int(round(next_width))][0]<255) :
        next_node[0] = next_width
        next_node[1] = next_height
        next_node[2] = next_angle
        visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] = 1
        return True, next_node
    else:
        return False, next_node


def action_minus_sixty(node,canvas,visited,step): # Local angles
    next_node = node.copy()
    
    next_angle = next_node[2] - 60
    if next_angle < 0:
        next_angle += 360
    
    next_angle %= 360 
    next_width = threshold(next_node[0] + step*np.cos(np.deg2rad(next_angle)))
    next_height = threshold(next_node[1] + step*np.sin(np.deg2rad(next_angle)))
    # print("Next Angle ",next_angle)
    # print("Width, Height: ",next_width,next_height)

    if (next_height>0 and next_height<canvas.shape[0]) and (next_width>0 and next_width<canvas.shape[1]) and (canvas[int(round(next_height))][int(round(next_width))][0]<255) :
        next_node[0] = next_width
        next_node[1] = next_height
        next_node[2] = next_angle
        visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] = 1
        return True, next_node
    else:
        return False, next_node

def action_plus_thirty(node,canvas,visited,step): # Local angles
    next_node = node.copy()
    next_angle = next_node[2] + 30
    if next_angle < 0:
        next_angle += 360 
    next_angle %= 360
    next_width = threshold(next_node[0] + step*np.cos(np.deg2rad(next_angle)))
    next_height = threshold(next_node[1] + step*np.sin(np.deg2rad(next_angle)))
    # print("Next Angle ",next_angle)
    # print("Width, Height: ",next_width,next_height)

    if (next_height>0 and next_height<canvas.shape[0]) and (next_width>0 and next_width<canvas.shape[1]) and (canvas[int(round(next_height))][int(round(next_width))][0]<255) :
        next_node[0] = next_width
        next_node[1] = next_height
        next_node[2] = next_angle
        visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] = 1
        return True, next_node
    else:
        return False, next_node

def action_plus_sixty(node,canvas,visited,step): # Local angles
    next_node = node.copy()
    next_angle = next_node[2] + 60
    if next_angle < 0:
        next_angle += 360
    next_angle %= 360
    next_width = threshold(next_node[0] + step*np.cos(np.deg2rad(next_angle)))
    next_height = threshold(next_node[1] + step*np.sin(np.deg2rad(next_angle)))
    # print("Next Angle ",next_angle)
    # print("Width, Height: ",next_width,next_height)

    if (next_height>0 and next_height<canvas.shape[0]) and (next_width>0 and next_width<canvas.shape[1]) and (canvas[int(round(next_height))][int(round(next_width))][0]<255) :
        next_node[0] = next_width
        next_node[1] = next_height
        next_node[2] = next_angle
        visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] = 1
        return True, next_node
    else:
        return False, next_node


def astar(initial_state,final_state,canvas,step):
    """
    @brief: This function implements the A* algorithm to find the path between given
    start node and goal node 
    :param initial_state: Start Node
    :param final_state: Final Node

    Open List is a heap queue which has the cost as the key to sort the heap
    Closed list is a dictionary which has key as the current node and value as the parent node
    This function is robust enough to give the no solution prompt for goal/start states in the obstacle space
    """
    open_list = []
    closed_list = {}
    back_track_flag = False
    visited_nodes =np.zeros((500,800,12))
    hq.heapify(open_list)
    hq.heappush(open_list,[0,initial_state,initial_state])
    node = hq.heappop(open_list)
    print(node)
    flag,n_state = action_plus_sixty(node[2],canvas,visited_nodes,step)
    # if(flag):
    print(flag,n_state)
    flag,n_state = action_plus_thirty(node[2],canvas,visited_nodes,step)
    # if(flag):
    print(flag,n_state)
    flag,n_state = action_zero(node[2],canvas,visited_nodes,step)
    # if(flag):
    print(flag,n_state)
    flag,n_state = action_minus_thirty(node[2],canvas,visited_nodes,step)
    # if(flag):
    print(flag,n_state)
    flag,n_state = action_minus_sixty(node[2],canvas,visited_nodes,step)
    # if(flag):
    print(flag,n_state)

    # while(len(open_list)>0):
    #     # 0: cost, 1: parent node, 2: present node
    #     node = hq.heappop(open_list)
    #     # print(node)
    #     closed_list[(node[2][0],node[2][1])] = node[1] #Converted to tuple because the key for dictionary should be immutable
    #     present_cost = node[0] #Present node cost to come
    #     if list(node[2]) == final_state: #Checks if the popped node is goal node
    #         back_track_flag = True
    #         print("Back Track") #Back tracking starts
    #         break #come out of the while loop

    # if(back_track_flag):
    #     #Call the backtrack function
    #     back_track(initial_state,final_state,closed_list,canvas)

    # else:
    #     print("Solution Cannot Be Found")


def back_track(initial_state,final_state,closed_list,canvas):
    """
    @brief: This function backtracks the start node after reaching the goal node.
    This function is also used for visualization of explored nodes and computed path using OpenCV.
    A stack is used to store the intermediate nodes while transversing from the goal node to start node.

    :param initial_state: Start Node
    :param final_state: Goal Node
    :param closed_list: Dictionary that contains nodes and its parents
    :param canvas: Canvas Image 
    """
    #Creating video writer to generate a video.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Dijkstra-KumaraRitvik-Oruganti.avi',fourcc,1000,(canvas.shape[1],canvas.shape[0]))
    
    keys = closed_list.keys() #Returns all the nodes that are explored
    path_stack = [] #Stack to store the path from start to goal
    for key in keys:
        canvas[key[1]][key[0]] = [255,255,255] #Denoting the explored nodes with white color
        cv2.imshow("Nodes Exploration",canvas)
        cv2.waitKey(1)
        out.write(canvas)
    parent_node = closed_list[tuple(final_state)]
    path_stack.append(final_state) #Appending the final state because of the loop starting condition
    while(parent_node!=initial_state):
        # print("Parent Node",parent_node)
        # canvas[parent_node[1]][parent_node[0]] = [19,209,158]
        path_stack.append(parent_node)
        parent_node = closed_list[tuple(parent_node)]
        # cv2.imshow("Path",canvas)
        # cv2.waitKey(1)
    cv2.circle(canvas,tuple(initial_state),3,(0,255,0),-1)
    cv2.circle(canvas,tuple(final_state),3,(0,0,255),-1)
    path_stack.append(initial_state) #Appending the initial state because of the loop breaking condition
    while(len(path_stack)>0):
        path_node = path_stack.pop()
        canvas[path_node[1]][path_node[0]] = [19,209,158]
        out.write(canvas)
    
    cv2.imshow("Nodes Exploration",canvas)
    out.release()

if __name__ == '__main__':
    start_time = time.time() #Gives the time at which the program has started
    canvas = np.ones((250,400,3),dtype="uint8") #Creating a blank canvas
    canvas = draw_obstacles(canvas) #Draw the obstacles in the canvas
    plt.imshow(canvas)
    plt.show()
    #Uncomment the below lines to see the obstacle space. Press Any Key to close the image window
    # cv2.imshow("Canvas",canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    initial_state,final_state,step = take_inputs(canvas) #Take the start and goal node from the user
    #Changing the cartesian coordinates to image coordinates:
    initial_state[1] = canvas.shape[0]-1 - initial_state[1]
    final_state[1] = canvas.shape[0]-1 - final_state[1]
    # print(initial_state,final_state,start_angle,goal_angle)

    astar(initial_state,final_state,canvas,step) #Compute the path using A Star Algorithm
    
    end_time = time.time() #Time taken to run the whole algorithm to find the optimal path
    cv2.waitKey(0) #Waits till a key is pressedy by the user
    cv2.destroyAllWindows() #destroys all opencv windows
    print("Code Execution Time: ",end_time-start_time) #Prints the total execution time