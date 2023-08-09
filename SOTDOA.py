# -*- coding: utf-8 -*-
"""
Created on Mon Aug 02 11:03:02 2023

Algoritmo SOTDOA

Se obtiene la estimación de la ubicación de una pisada dentro de una habitación.


En este programa se debe definir la ubicación de los 5 sensores en el arreglo 'posicion'


Se emulan pisadas cada 0.25 cm a lo largo de la habitación. 
La ubicación de las pisadas se guardan en el arreglo 'P'


El programa calcula el error de estimación para cada pisada, 
así como el MSE y el RMSE para todas las pisadas 


@authors: Luis Sánchez Márquez

"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.system('cls')

posicion = np.array([0.20, 0.50, 2.27, 0.50, 2.27, 3.00, 0.20, 3.00, 1.25, 1.75]) #[X1 Y1 X2 Y2 X3 Y3 X4 Y4 X5 Y5]
#posicion = np.array([2.04, 1.56, 1.80, 3.10, 1.20, 1.53, 0.20, 1.54, 1.82, 0.30])

NumS = 5 # Número de sensores

# Asignando las coordenadas de los sensores
S1 = np.array([posicion[0], posicion[1]]) #[0.1,0.1]
S2 = np.array([posicion[2], posicion[3]]) #[2.9,0.1];
S3 = np.array([posicion[4], posicion[5]]) #[2.9,2.9];
S4 = np.array([posicion[6], posicion[7]]) #[0.1,2.9];
S5 = np.array([posicion[8], posicion[9]]) #[1.5,1.5];

S  = np.array([S1, S2, S3, S4, S5])

# temp = S2[0]    # S1[0, 1] = 0.1
# temp2 = S[3,1]  # S[3,1] = 2.9

# dimensiones de la habitación 
#si la habitación es de 2.5 m x 3.3 m, entonces Linf = [0,0]  y LSup = [2.5,3.3]

LInf = np.array([0.2, 0.2]) # x1, y1
LSup = np.array([2.5, 3.3]) # x2, y2

# Coordenadas (x,y) de los vertices de las regiones
st = 0.1
NumR = int((np.round((LSup[1])/st)+1)*(np.round((LSup[0])/st)+1))

# Calculo de los centroides de las regiones
C0=0
k=0
l=0
CR = np.zeros((NumR,2))

for i in range (0, NumR, 1): #1:NumR
    CR[i,0] = C0 + l
    CR[i,1] = C0 + k
    k = k + st
    if k >= LSup[1] + st:
        k = 0
        l = l + st

# Distancias entre sensores(i) y centroides de las regiones(j)
D = np.zeros((NumS,NumR))
for i in range(0,NumS,1): #1:NumS
    for j in range(0,NumR,1): # j = 1:NumR
        D[i,j] = np.sqrt(np.power((S[i,0]-CR[j,0]),2) + np.power((S[i,1]-CR[j,1]),2))

# Número de elementos del vector caracteristico
NumEC = int((NumS*(NumS-1))/2)

# Vectores caracteristicos de las regiones
Z = np.zeros((NumR,NumEC))

for i in range(0,NumR,1): #1:NumR
    Z[i,0]  = np.sign(D[0,i]-D[1,i]);
    Z[i,1]  = np.sign(D[0,i]-D[2,i]);
    Z[i,2]  = np.sign(D[1,i]-D[2,i]);
    Z[i,3]  = np.sign(D[0,i]-D[3,i]);
    Z[i,4]  = np.sign(D[1,i]-D[3,i]);
    Z[i,5]  = np.sign(D[2,i]-D[3,i]);
    Z[i,6]  = np.sign(D[0,i]-D[4,i]);
    Z[i,7]  = np.sign(D[1,i]-D[4,i]);
    Z[i,8]  = np.sign(D[2,i]-D[4,i]);
    Z[i,9] =  np.sign(D[3,i]-D[4,i]);

step = 0.25; #Para pisadas cada 25cm
NumP = int((np.round(LSup[1]-step)/step)*np.round((LSup[0]-step)/step))
P0 = 0.25
k = 0
l = 0

P = np.zeros((NumP,2))

# Se emulan pisadas cada 0.25 cm a lo largo de la habitación
for i in range(0,NumP,1): #1:NumP
    P[i,0] = P0 + l
    P[i,1] = P0 + k
    k = k + step
    if k >= int(np.round(LSup[1]-step)):
        k = 0
        l = l + step

#########################   Otra trayectoria propuesta   #########################
Ucirc_1 = np.array([2.25, 1.75])
Ucirc_2 = np.array([1.95, 2.45])
Ucirc_3 = np.array([1.25, 2.75])
Ucirc_4 = np.array([0.55, 2.45])
Ucirc_5 = np.array([0.25, 1.75])
Ucirc_6 = np.array([0.55, 1.05])
Ucirc_7 = np.array([1.25, 0.75])
Ucirc_8 = np.array([1.95, 1.05])

Ucirc = np.array([Ucirc_1,Ucirc_2,Ucirc_3,Ucirc_4,Ucirc_5,Ucirc_6,Ucirc_7,Ucirc_8])
P = Ucirc 
NumP = int(np.size(P)/2)

##########################  Comentar hasta aquí ##################################


# TOA entre sensores(i) y pisadas(j)
T = np.zeros((NumS,NumP))
D1 = np.zeros ((NumS,NumP))
vel_cte = 1000 # Velocidad de propagación de la vibración sobre el concreto

for i in range(0,NumS,1): # 1:NumS
    for j in range(0,NumP,1): # 1:NumP
        D1[i,j] = np.sqrt(np.power((S[i,0]-P[j,0]),2) + np.power((S[i,1]-P[j,1]),2))
        
T = D1/vel_cte



# Vectores caracteristicos de las pisadas
Y = np.zeros ((NumP,NumEC))

for i in range(0,NumP,1): #1:NumP

    Y[i,0]  = np.sign(T[0,i]-T[1,i]);
    Y[i,1]  = np.sign(T[0,i]-T[2,i]);
    Y[i,2]  = np.sign(T[1,i]-T[2,i]);
    Y[i,3]  = np.sign(T[0,i]-T[3,i]);
    Y[i,4]  = np.sign(T[1,i]-T[3,i]);
    Y[i,5]  = np.sign(T[2,i]-T[3,i]);
    Y[i,6]  = np.sign(T[0,i]-T[4,i]);
    Y[i,7]  = np.sign(T[1,i]-T[4,i]);
    Y[i,8]  = np.sign(T[2,i]-T[4,i]);
    Y[i,9] =  np.sign(T[3,i]-T[4,i]);

# Y = hardlims(Y); %Quité


# Calculando distancia de Hamming
A = np.zeros((NumR,NumEC))
H = np.zeros((NumP,NumR))

for p in range (0,NumP,1): # 1:NumP
    for i in range (0,NumR,1): # 1:NumR
        for j in range(0,NumEC,1): # 1:NumEC
            A[i,j] = (Y[p,j] != Z[i,j])
        H[p,i] = np.sum(A[i])

# Estimación de las pisadas
E = np.zeros((NumP,2))
ka_p = np.zeros((NumP,1))
k_p = np.zeros((NumP,4))

for j in range(0,NumP,1): # 1:NumP
    ka = np.min(H[j]) #Encuentra el mínimo (Agregué)
    ka_p[j] = ka
    k = np.argwhere(H[j] == ka) #Encuentra los índices de los valores que tienen el mínimo encontrado
    [m,n] = np.shape(k) #Para saber el número de mínimos encontrados
    Ex = 0; Ey = 0
    for i in range(0,m,1):# 1:m
        Ex = Ex + CR[k[i],0]
        Ey = Ey + CR[k[i],1]
    E[j,0] = Ex/m
    E[j,1] = Ey/m

Error = np.zeros(NumP)
# Error de posiciÃ³n entre pisadas y estimaciones
for i in range(0,NumP,1): # 1:NumP
    Error[i] = np.sqrt(np.power((P[i,0]-E[i,0]),2) + np.power((P[i,1]-E[i,1]),2))
    
sqe = np.power(Error,2)
mse = np.sum(sqe)/NumP
rmse = np.sqrt(mse)


fig, f1 = plt.subplots()
f1.plot(S[:,0], S[:,1], 'ro', ms = 10, label = "Sensores")
f1.plot(P[:,0], P[:,1], 'b-.', marker = 'o',ms = 8, label = "Real")
f1.plot(E[:,0], E[:,1], 'ms--', ms = 8, label = "Estimada")
plt.title('Algoritmo SO-TDOA')
plt.xlabel('Distancia, m')
plt.ylabel('Distancia, m')
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.1)
plt.legend(bbox_to_anchor =(0.75, 1.15), ncol = 3)