import cv2
import sys

import numpy as np
import maxflow
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import img_as_ubyte
from dijkstar import Graph, find_path
import os
import preprocess_minh
import matplotlib.pyplot as plt
import numpy as np
import example.example as eb
import copy
import os



def eu_dis(v1, v2):
    return np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2)



def computeEnerge(label, fg, bg1,  neighbor, lamda,  LAB_map, sigma1):
    return (giveDataEnerge(label, fg, bg1) + giveSmoothEnerge(label, neighbor, lamda,  LAB_map, sigma1))

def giveDataEnerge( label, fg, bg1):
    energe = 0
    h,w = fg.shape

    for x in range (h):
        for y in range (w):
            if label[x][y] == 1:
                energe += fg[x][y]
            elif label[x][y] == 0:
                energe += bg1[x][y]

    return energe


def giveSmoothEnerge(label, neighbor, lamda,  LAB_map, sigma1 ):  # compute SmoothEnerge
    energe = 0
    h,w = label.shape
    for x in range (h):
        for y in range (w):
            u = x*w + y
            for i in range (4):
                a = x + neighbor[i][0]
                b = y + neighbor[i][1]
                if (a >= 0 and a <h and b >= 0 and b < w):
                    v = a*w + b
                    if v < u:
                        continue
                    if label[x][y] == label[a][b]:
                        continue
                    energe += lamda * np.e ** (-(eu_dis(LAB_map[x][y], LAB_map[a][b]) ** 2) / sigma1)
    return energe


src_folder     = '/media/minh/DATA/Study/database/Interative_Dataset/images/images/'
label_folder   = '/media/minh/DATA/Study/database/Interative_Dataset/images-labels/images-labels/'
output_folder  = '/media/minh/DATA/Study/Results/segmentation_scribble/Segment_2020/Dahu_MRF_color_distribution/markov/'
score_folder = '/media/minh/DATA/Study/Results/segmentation_scribble/Segment_2020/Dahu_MRF_color_distribution/score/'
fg_folder = '/media/minh/DATA/Study/Results/segmentation_scribble/Segment_2020/Dahu_MRF_color_distribution/fg/'
bg_folder = '/media/minh/DATA/Study/Results/segmentation_scribble/Segment_2020/Dahu_MRF_color_distribution/bg/'
initial_folder = '/media/minh/DATA/Study/Results/segmentation_scribble/Segment_2020/Dahu_MRF_color_distribution/initial/'
output_post_folder  = '/media/minh/DATA/Study/Results/segmentation_scribble/Segment_2020/Dahu_MRF_color_distribution/post/'

input_file = os.listdir(src_folder)

print(len(input_file))

for entry in input_file:
    
    print(entry)

    # if entry == '189080.jpg':



    parts = entry.split(".")
    

    src_name = src_folder + entry
    label_name  = label_folder  + parts[0] + '-anno.png'




    img = cv2.imread(src_name)
    label_gray = cv2.cvtColor(cv2.imread(label_name), cv2.COLOR_BGR2GRAY)
    ima_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)


    h, w = label_gray.shape
    print(h, w)





    ###### LAB map

    LAB_map_raw = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    LAB_map = np.zeros_like(LAB_map_raw, dtype=np.int8)
    for i in range(len(LAB_map)):
        for j in range(len(LAB_map[0])):
            LAB_map[i, j][0] = LAB_map_raw[i, j][0] / 255 * 100
            LAB_map[i, j][1] = LAB_map_raw[i, j][1] - 128
            LAB_map[i, j][2] = LAB_map_raw[i, j][2] - 128



    ##### confidence map



    list_bg= []
    list_fg= []

    f_fg = np.zeros((h,w))
    f_bg = np.zeros((h,w))

    for i in range (0, h):
        for j in range (0, w):
            if (label_gray[i][j] > 10 and label_gray[i][j] < 100 ):
                list_bg.append(ima_lab[i][j])
                f_bg[i][j] = 255
                
            if (label_gray[i][j] > 100):
                list_fg.append(ima_lab[i][j])
                f_fg[i][j] = 255

    list_bg = np.asarray(list_bg)
    list_fg = np.asarray(list_fg)





    score, score1 = preprocess_minh.preprocess_postproba(ima_lab, list_fg, list_bg)
    score = np.array(score * 255, dtype="uint8") # convert to uint8


    height = h*2-1;
    width = w*2-1;

    F_fg = cv2.resize(f_fg, (width,height))
    F_fg = np.array(F_fg, dtype="uint8") # convert to uint8

    F_bg = cv2.resize(f_bg, (width,height))
    F_bg = np.array(F_bg, dtype="uint8") # convert to uint8

    print(F_fg.shape)

    dmap_scalar_fg = eb.dahu_scribble(img, score,  F_fg)
    dmap_scalar_bg = eb.dahu_scribble(img, score,  F_bg)

    fg = np.zeros((h,w))
    bg1 = np.zeros((h,w))

    for i in range(0,h):
        for j in range(0,w):
            fg[i][j] = dmap_scalar_fg[2*i][2*j]/255
            bg1[i][j] = dmap_scalar_bg[2*i][2*j]/255





    #### compute sigma

    neighbor = [[0,1],[0,-1],[1,0],[-1,0]]

    sigma1 = 0

    for x in range (h):
        for y in range (w):

            u = x*w + y
            for i in range (4):
                a = x + neighbor[i][0]
                b = y + neighbor[i][1]

                if (a >= 0 and a <h and b >= 0 and b < w):
                    v = a*w + b

                    if (v > u):
                        if sigma1 < eu_dis(LAB_map[x][y], LAB_map[a][b]):
                            sigma1 = eu_dis(LAB_map[x][y], LAB_map[a][b])





    sigma1 = sigma1 ** 2 * 1

    print("sigma1 = " + str(sigma1))


    ############ Graphcut

    lamda  = 0.2


    label = np.zeros((h,w))
    label_ini = np.zeros((h,w))

    label[fg<=bg1] = 1
    label_ini[fg<=bg1] = 1

    oldEnergy = computeEnerge(label, fg, bg1,  neighbor, lamda,  LAB_map, sigma1)
    print(oldEnergy)





    nodes = []
    edges = []

    cap_source = fg
    cap_sink = bg1




    for x in range (h):
        for y in range (w):
            u = x*w + y
            nodes.append((u, cap_source[x][y] , cap_sink[x][y]))

    # print(u, reflect.index(u))




    for x in range (h):
        for y in range (w):
            u = x*w + y        
            for i in range (4):
                a = x + neighbor[i][0]
                b = y + neighbor[i][1]
                if (a >= 0 and a <h and b >= 0 and b < w):
                    v = a*w + b
                    if (v > u):
                        weight = lamda * np.e ** (-(eu_dis(LAB_map[x][y], LAB_map[a][b]) ** 2) / sigma1)
                        edges.append((u, v, weight))    


    ####GraphCuts####
    g = maxflow.Graph[float](len(nodes), len(edges))

    nodelist = g.add_nodes(len(nodes))
    for node in nodes:
        g.add_tedge(node[0], node[1], node[2])

    for edge in edges:
        g.add_edge(edge[0], edge[1], edge[2], edge[2])

    flow = g.maxflow()  

    for vect in nodes:
        v = vect[0]
        if g.get_segment(v) == 0:  # beta
            x = int(np.floor(v/w))
            y = v%w
            label[x][y] = 0
        else:  # alpha
            label[x][y] = 1

    newEnergy = computeEnerge(label, fg, bg1,  neighbor, lamda,  LAB_map, sigma1)
    print(newEnergy)

    #  Post processing

    label_gray[label_gray>100] = 255

    label = np.array(label, dtype="uint8") # convert to uint8
    # Find largest contour in intermediate image so that it contains the markers 
    cnts, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    con = []
    for i in range(len(cnts)):
        biggest = np.zeros(label.shape, np.uint8)
        cv2.drawContours(biggest, [cnts[i]], -1, 255, cv2.FILLED)
        biggest = cv2.bitwise_and(label_gray, biggest)
        if (np.sum(biggest) > 0):
            con.append(cnts[i])

    print(len(con))

    biggest = np.zeros(label.shape, np.uint8)

    if (len(con) != 0):
        cnt = max(con, key=cv2.contourArea)

        # Output            
        cv2.drawContours(biggest, [cnt], -1, 255, cv2.FILLED)

    print(np.max(biggest))
    biggest = np.array(biggest, dtype="uint8") # convert to uint8
    output_name = output_post_folder + entry
    cv2.imwrite(output_name, biggest)

    label = np.array(label*255, dtype="uint8") # convert to uint8
    output_name = output_folder + entry
    cv2.imwrite(output_name, label)

    score_name = score_folder + entry
    cv2.imwrite(score_name, score)


    fg = np.array(fg*255, dtype="uint8") # convert to uint8
    fg_name = fg_folder + entry
    cv2.imwrite(fg_name, fg)    

    bg1 = np.array(bg1*255, dtype="uint8") # convert to uint8
    bg_name = bg_folder + entry
    cv2.imwrite(bg_name, bg1)        

    label_ini = np.array(label_ini*255, dtype="uint8") # convert to uint8
    initial_name = initial_folder + entry
    cv2.imwrite(initial_name, label_ini)   


