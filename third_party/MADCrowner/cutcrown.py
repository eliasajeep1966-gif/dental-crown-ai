#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量裁剪 crown mesh
"""

import os
import time
from pathlib import Path
import trimesh
# This file is classified as condidential level C3 with SenseTime Group.
import trimesh
import scipy.spatial
import numpy as np
import scipy.spatial
import scipy.misc
import networkx as nx
import scipy.interpolate
import time
__all__ = ['get_crown']

def standard_fit(X):
    # Find the average of points (centroid) along the columns
    C = np.average(X, axis=0)
    # Create CX vector (centroid to point) matrix
    CX = X - C
    # Singular value decomposition
    U, S, V = np.linalg.svd(CX)
    # The last row of V matrix indicate the eigenvectors of
    # smallest eigenvalues (singular values).
    N = V[-1]
    N = N/np.linalg.norm(N)
    return C, N

def sq1(p1,p2,p3):
    a = np.linalg.norm(p1-p2)
    b = np.linalg.norm(p3-p2)
    c = np.linalg.norm(p3-p1)
    p = (a+b+c)/2
    s1 = p*(p-a)*(p-b)*(p-c)
    return(s1**0.5)

def neiToGroup(dicNei1):
    nodeOnB = sorted(list(dicNei1.keys()))
    dicVisited = {x:False for x in nodeOnB}
    groups = [] 

    while True:
        list1 = []
        for x in nodeOnB:
            if not dicVisited[x]:
                list1.append(x)
                dicVisited[x] = True
                break
        if len(list1) ==0:
            break

        i1 = 0

        while i1 < len(list1):
            a1 = list1[i1]
            for x in dicNei1[a1]:
                if dicVisited[x]:
                    continue
                list1.append(x)
                dicVisited[x] = True
                
            i1+=1
            
        groups.append(list1)
    return(groups)
    
def subNei(edges2, set1):
    dicNei1 = {}
    
    for x in edges2:
        if x[0] not in set1 or x[1] not in set1:
            continue
        for a,b in [x,x[::-1]]:
            
            if a not in dicNei1:
                dicNei1[a] = [b]
            else:
                if b not in dicNei1[a]:
                    dicNei1[a].append(b)
    return(dicNei1)
    
def getEdgesOnOpenings(faceInput):
    dic1 = {}
    for f1 in faceInput:
        for j1 in range(3):
            k1,k2 = sorted([f1[x] for x in range(3) if x!=j1])
            if (k1,k2) in dic1:
                dic1[(k1,k2)] +=1
            else:
                dic1[(k1,k2)] = 1
                
    edgesOnB = [x for x in dic1 if dic1[x] ==1]  
    return(edgesOnB)

def getLargestCircle(setBottom0, edges0, ver0):
    G_bottom = nx.Graph()
    edgeListB = [x for x in edges0 if x[0] in setBottom0 and x[1] in setBottom0]
    G_bottom.add_edges_from(edgeListB)
    cycleList_b  = nx.algorithms.cycles.cycle_basis(G_bottom)

    cycleList_b_len = []
    for c1 in cycleList_b:
        sum1 = 0
        for i1 in range(len(c1)):
            j1 = (i1+1)%len(c1)
            
            sum1 += np.linalg.norm(np.array(ver0[c1[i1]]) - np.array(ver0[c1[j1]]))
        cycleList_b_len.append(sum1)    

    bottomCircle0 = cycleList_b[np.argmax(cycleList_b_len)]
    return(bottomCircle0)


def get_boundary(mesh:trimesh.Trimesh):
    outline = mesh.outline()
    max_length = 0
    boundary_idx = None
    for p in outline.paths:
        entities = outline.entities[p]
        ee = []
        for e in entities:
            ee.extend(e.points)
        if max_length>len(ee):
            continue
        else:
            max_length = len(ee)
            boundary_idx = ee
    boundary_idx = boundary_idx[:-1]
    return boundary_idx

###
###

def get_crown(mesh, mesh_nl):

    ## 通过提取颈缘线的中心与法向量，建立球坐标系，从而拟合出一个相对平滑的颈缘线平面表达，从而分割牙冠mesh
    mesh1 = mesh_nl
    meshCrown = mesh

    bottomeEdges = getEdgesOnOpenings(mesh1.faces)
    setBottom = set([x[0] for x in bottomeEdges] + [x[1] for x in bottomeEdges])
    ## 获取颈缘线 
    bottomCircle  = getLargestCircle(setBottom, mesh1.edges, mesh1.vertices)
    lenB = len(bottomCircle)

    ### 获取颈缘线的中心与法向量，后面以此建立球坐标系
    cB, nB = standard_fit(np.array([mesh1.vertices[x]   for x in bottomCircle ]))

    ## 确保法向量 nB 从下向上
    meanProj = np.mean([np.sum((np.array(x) - cB)*nB) for x in mesh1.vertices ])
    if meanProj<0:
        nB = -nB

    ## 确保颈缘线上点的旋转方向    
    sign1 = np.sign(np.sum(nB* np.cross( np.array(mesh1.vertices[bottomCircle[lenB//6]]) - cB, np.array(mesh1.vertices[bottomCircle[0]]) - cB)))

    if sign1 < 0:
        bottomCircle = bottomCircle[::-1]
    
    
    ##排除掉少量逆行的点
    # print(sign1)
    #sys.exit()
    bottomCircle2 = [bottomCircle[0]]
    for k1 in range(1,lenB):
        i1 = bottomCircle[k1]
        j1 = bottomCircle2[-1]
        a1 =  np.cross( np.array(mesh1.vertices[i1]) - cB, np.array(mesh1.vertices[j1]) - cB)
        a1 = a1/np.linalg.norm(a1)
        #print(np.sum(nB*a1))
        
        if np.sum(nB*a1)>0.2:
            bottomCircle2.append(i1)
    
    # print(len(bottomCircle), len(bottomCircle2))   
    bottomCircle = [bottomCircle2[-1]] +  bottomCircle2[:-1]

    bottomCircle2 = [bottomCircle[0]]
    for k1 in range(1,lenB):
        i1 = bottomCircle[k1]
        j1 = bottomCircle2[-1]
        a1 =  np.cross( np.array(mesh1.vertices[i1]) - cB, np.array(mesh1.vertices[j1]) - cB)
        a1 = a1/np.linalg.norm(a1)
        #print(np.sum(nB*a1))   
        #sign2 = int(np.sign(np.sum(nB* np.cross( np.array(mesh1.vertices[i1]) - cB, np.array(mesh1.vertices[j1]) - cB))))
        if np.sum(nB*a1) > 0.4 :
            bottomCircle2.append(i1)
            break
    bottomCircle2 += bottomCircle[k1+1:]


    ## 找出颈缘线上各点在球坐标系中的 旋转角（phi ，0~360） 跟与Z轴的夹角（theta 0~180）
    bottomVect2 = [np.array(mesh1.vertices[x]) - cB for x in bottomCircle2]
    v0 = bottomVect2[0]
    v0 = v0/np.linalg.norm(v0)
    eps1 = 1e-32
    bottomPhi_head = [0]
    bottomPhi = [np.arccos(np.clip(np.sum(x*v0)/np.linalg.norm(x), -1+eps1, 1-eps1))*180/np.pi if np.sum(nB*np.cross(x,v0))>=0 else 360 - np.arccos(np.sum(x*v0)/np.linalg.norm(x))*180/np.pi  for x in bottomVect2[1:]]
    bottomPhi = bottomPhi_head + bottomPhi
    bottomPhi = [x%360 for x in bottomPhi]
    bottomTheta = [np.arccos(np.sum(x*nB)/np.linalg.norm(x)) *180/np.pi for x in bottomVect2]
    # bottomPhi2 = bottomPhi + [bottomPhi[0]+360]
    # bottomTheta2 = bottomTheta + [bottomTheta[0]]


    ## 拟合 颈缘线上的 phi 角与 theta 角的对应关系   theta = f(phi)
    ## 通过调整这一步的参数，可以调整获得的颈缘线平面度平滑度

    
    minStp = 1.5 ## minSt越大，结果约平滑
    stps = [0]
    for i1 in range(1,len(bottomPhi)):
        if bottomPhi[0]+360 - bottomPhi[i1]<minStp:
            break
        if bottomPhi[i1] - bottomPhi[stps[-1]] >= minStp:
            stps.append(i1)
        
    # print(len(bottomCircle), len(bottomCircle2))
    # print(len(bottomPhi))
    # print(len(stps))
    
    interP1 = scipy.interpolate.InterpolatedUnivariateSpline([bottomPhi[x ] for x in stps] + [bottomPhi[0]+360], [bottomTheta[x]  for x in stps] + [bottomTheta[0]],k=3)   # 改变参数k的值，可能可以调节平滑度


    ### 求出牙冠mesh上各个点的 phi 与theta 角，  theta < f(phi)的，就是颈缘线平面上方的点
    vectCrown = [np.array(x)-cB for x in meshCrown.vertices] 
    crownPhi = [np.arccos(np.sum(x*v0)/np.linalg.norm(x))*180/np.pi if np.sum(nB*np.cross(x,v0))>=0 else 360 - np.arccos(np.sum(x*v0)/np.linalg.norm(x))*180/np.pi  for x in vectCrown] 
    crownPhi = [x%360 for x in crownPhi]
    crownTheta = np.array([np.arccos(np.sum(x*nB)/np.linalg.norm(x)) *180/np.pi for x in vectCrown])
    crownThetaFit = interP1(crownPhi)

    setUp1 = set(np.where(crownTheta<=crownThetaFit)[0].tolist())
    setDown1 = set( [x for x in range(meshCrown.vertices.shape[0]) if x not in setUp1 ] )

    ### 如果一个edge的两端，一个在颈缘线平面上方，一个在颈缘线平面下方，就把下方的点沿着这个edge，移动到颈缘线平面上。 （一般这样的点会对应多个edge，只移动一次） 
    dicDown1 = {}
    for e1 in meshCrown.edges:
        if e1[0] in setDown1 and e1[1] in setUp1:
            dicDown1[e1[0]] = e1.tolist()
        
        elif e1[1] in setDown1 and e1[0] in setUp1:
            dicDown1[e1[1]] = e1.tolist()

    for i1 in dicDown1:
        j1 = [x for x in dicDown1[i1] if x!=i1][0]
        p_i1 = crownTheta[i1]
        p_i1_fit = crownThetaFit[i1]
        
        p_j1 = crownTheta[j1]
        p_j1_fit = crownThetaFit[j1]
        
        down_i  = p_i1 - p_i1_fit
        up_j = p_j1 -  p_j1_fit
        pNew = ( abs(up_j)*meshCrown.vertices[i1] + abs(down_i)*meshCrown.vertices[j1]) / (abs(down_i) + abs(up_j))
        meshCrown.vertices[i1] = pNew

    setUp2 = set(list(setUp1) + list(dicDown1.keys()))    
        # if not (up_j<0 and down_i>0):
            # print('error',i1,j1,up_j,down_i)
            # continue
        
    # sys.exit()
        
            
            
    # setUp1 = set(np.where(crownTheta<crownThetaFit)[0].tolist())


    setC = setUp2
    facesS = [x for x in  range(meshCrown.faces.shape[0])  if meshCrown.faces[x,1] in setC and   meshCrown.faces[x,0] in setC and  meshCrown.faces[x,2] in setC]
    out1 = meshCrown.submesh([facesS], append=True)

    return out1
# ========= 参数 ========= #
root_vis   = Path("/mnt/disk1/linda/CrownMVM/test_outputs/diffusion_sdf")
root_clean = Path("/mnt/disk1/linda/clean_mesh_with_margin")
out_dir    = root_vis / "cut"
out_dir.mkdir(parents=True, exist_ok=True)


