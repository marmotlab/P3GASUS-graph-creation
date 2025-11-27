import numpy as np
import cv2
import math
from matplotlib.colors import hsv_to_rgb
import imageio
import colorsys
def make_gif(images, file_name):
    """record gif"""
    imageio.mimwrite(file_name, images, subrectangles=True)
    print("wrote gif")

def getFreeCell(world):
    
    listOfFree = np.swapaxes(np.where(world==0), 0,1)
    if(len(listOfFree)==0):
        raise Exception("No more free Cells")

    np.random.shuffle(listOfFree)
    return (listOfFree[0][0], listOfFree[0][1])

def generateWarehouse(num_block=[-1,-1], length=40, shelfSize=5,lbRatio=2/3,freeSpaceRatio=1/3, shelfWidth=1):
    if(num_block[0]!=-1):
        length = np.random.randint(low=num_block[0], high=num_block[1]+1)
    breadth = int(length/lbRatio)
    world = np.zeros((length,breadth), dtype='int64')
    # Generate shelves
    noShelves = int((breadth*(1-freeSpaceRatio))/(shelfSize+1))
    freeSpace = int((breadth - noShelves*(shelfSize+1))/2)
    for i in range(freeSpace, freeSpace+noShelves*(shelfSize+1), shelfSize+1):
        for j in range(1, shelfWidth+1):
            for k in range(j, length-1, shelfWidth+1):
                world[k,i:i+shelfSize] = -1
    return world

    
def int_to_rgba(value: int):
    """the colors of agents and goals"""
    c = {}
    c[0] = [1,1,1]
    c[-1] = [0,0,0]
    c[-2] = [0.5,0.5,0.5]
    
    if(value in c):
        return c[value]
    
    hue = (value * 0.618033988749895) % 1  # Using the golden ratio to generate well-distributed hues
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)  # High saturation and value for vibrant colors
    rgba = (*rgb, 1.0)
    return rgba[:3]

def getArrowPoints(direction, coord, scale, tailWidth, headWidth):

    if(np.array_equal(direction, np.array([0,1]))):
        halfScale = int(scale/2)-1
        center = [coord[1]*scale+halfScale, coord[0]*scale+halfScale]
        tailHeight = halfScale-2
        
        arrow = [
                [center[0], center[1]-tailWidth],
                [center[0]-tailHeight, center[1]-tailWidth], 
                [center[0]-tailHeight, center[1]+tailWidth],
                [center[0], center[1]+tailWidth],
                [center[0], center[1]+headWidth], 
                [center[0]+headWidth, center[1]], 
                [center[0], center[1]-headWidth]
                ]
        
    elif(np.array_equal(direction, np.array([1,0]))):
        halfScale = int(scale/2)-1
        center = [coord[1]*scale+halfScale, coord[0]*scale+halfScale]
        tailHeight = halfScale-2
        arrow = [
                [center[0]-tailWidth, center[1]],
                [center[0]-tailWidth, center[1]-tailHeight], 
                [center[0]+tailWidth, center[1]-tailHeight], 
                [center[0]+tailWidth, center[1]],
                [center[0]+headWidth, center[1]], 
                [center[0], center[1]+headWidth], 
                [center[0]-headWidth, center[1]]
                ]
        
    elif(np.array_equal(direction, np.array([0,-1]))):
        halfScale = int(scale/2)-1
        center = [coord[1]*scale+halfScale, coord[0]*scale+halfScale]
        tailHeight = halfScale-2
        arrow = [
                [center[0], center[1]+tailWidth],
                [center[0]+tailHeight, center[1]+tailWidth], 
                [center[0]+tailHeight, center[1]-tailWidth],
                [center[0], center[1]-tailWidth],
                [center[0], center[1]-headWidth], 
                [center[0]-headWidth, center[1]], 
                [center[0], center[1]+headWidth]
                ]
        
    elif(np.array_equal(direction, np.array([-1,0]))):
        halfScale = int(scale/2)-1
        center = [coord[1]*scale+halfScale, coord[0]*scale+halfScale]
        tailHeight = halfScale-2
        arrow = [
                [center[0]+tailWidth, center[1]],
                [center[0]+tailWidth, center[1]+tailHeight], 
                [center[0]-tailWidth, center[1]+tailHeight], 
                [center[0]-tailWidth, center[1]],
                [center[0]-headWidth, center[1]], 
                [center[0], center[1]-headWidth], 
                [center[0]+headWidth, center[1]]
                ]
    else:
        arrow = [] 
    return np.array(arrow, dtype='int64')

def getRectPoints(coord, scale):
    base = [coord[1]*scale, coord[0]*scale]
    return np.array([base, [base[0]+scale-1, base[1]], [base[0]+scale-1,base[1]+scale-1], [base[0], base[1]+scale-1]])    

def pixelForText(coord, scale):
    base = [coord[1]*scale, coord[0]*scale]
    return (int(math.floor(base[0]+scale*1/4)), int(math.floor(base[1]+scale*3/4)))


def getCenter(coord, scale):
    base = [coord[1]*scale, coord[0]*scale]
    return (int(math.floor(base[0]+scale/2)), int(math.floor(base[1]+scale/2)))

def getTriPoints( coord, scale):
    base = [coord[1]*scale, coord[0]*scale]
    return  np.array([[int(math.floor(base[0]+scale/2)), base[1]], [base[0]+scale-1,base[1]+scale-1], [base[0], base[1]+scale-1]])    

def renderWorld(scale=20, world = np.zeros(1),agents=[], goals=[], human=(-1,-1), humanPath=list(), zoneCoords = None, numAgents = 10):
    size = world.shape
    
    path = humanPath
        
    screen_height = scale*size[0]
    screen_width = scale*size[1]

    scene = np.zeros([screen_height, screen_width, 3])

    for coord,val in np.ndenumerate(world):
        cv2.fillPoly(scene, pts=[getRectPoints(coord=coord, scale=scale)], color=int_to_rgba(val))

    if(zoneCoords is not None):
        safetyZones = np.copy(scene)

        for coord in set(zip(zoneCoords[0][0], zoneCoords[0][1])):
                cv2.fillPoly(safetyZones, pts=[getRectPoints(coord=coord, scale=20)], color=[1,0,0])
        for coord in set(zip(zoneCoords[1][0], zoneCoords[1][1])):
                cv2.fillPoly(safetyZones, pts=[getRectPoints(coord=coord, scale=20)], color=[1,0.4,0.1])
        for coord in set(zip(zoneCoords[2][0], zoneCoords[2][1])):
                cv2.fillPoly(safetyZones, pts=[getRectPoints(coord=coord, scale=20)], color=[0,0,1])

        scene = cv2.addWeighted(safetyZones, 0.5, scene, 0.5, 0)
        

    for idx, val in enumerate(path):
        if(idx==len(path)-1):
            cv2.fillPoly(scene, pts=[drawStar(coord=val, scale=scale, diameter=scale, numPoints=5)], color=int_to_rgba(-2))
        else:
            direction = np.subtract(path[idx+1],val)
            # print(direction)
            arrowPoints = getArrowPoints(direction=direction, coord=val, scale=scale,tailWidth=scale/10,headWidth=scale/2-2)
            if len(arrowPoints)!=0:
                cv2.fillPoly(scene, pts=[arrowPoints], color=int_to_rgba(-2))

    for val,coord in enumerate(goals):
        cv2.circle(scene, getCenter(coord=coord, scale=scale), math.floor(scale/2)-1, int_to_rgba(val+1), -1)
        cv2.putText(scene, str(val+1), pixelForText(coord, scale), cv2.FONT_HERSHEY_SIMPLEX,scale/(40*(int(np.log10(val+1))+1)), (0,0,0), int(scale/20))


    for val,coord in enumerate(agents):
        cv2.fillPoly(scene, pts=[getRectPoints(coord=coord, scale=scale)], color=int_to_rgba(val+1))
        cv2.putText(scene, str(val+1), pixelForText(coord, scale), cv2.FONT_HERSHEY_SIMPLEX,scale/(40*(int(np.log10(val+1))+1)), (0,0,0), int(scale/20))

    if human[0]>=0 and human[1]>=0:
        cv2.fillPoly(scene, pts=[getTriPoints(coord=human, scale=scale)], color=int_to_rgba(-2))

    scene = scene*255
    scene = scene.astype(dtype='uint8')
    return scene