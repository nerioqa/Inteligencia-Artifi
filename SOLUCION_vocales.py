# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:15:59 2020
@ NERIO QUISPE ANCCO
"""
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pandas as pd
carpeta='C:/Users/NERIO-QA/Documents/ReconocimientoPatrones/PROYECTO RECONOCIMEINTO VOCALES/VOCALES/imagenes'
contenido=os.listdir(carpeta)

with open('dataset.csv', 'w',newline='', encoding='utf-8') as csvfile:
    writer=csv.writer(csvfile)
    for fichero in contenido:
        if os.path.isfile(os.path.join(carpeta, fichero)) and (fichero.endswith('.jpeg') or fichero.endswith('.jpg')): # LA LINEA ANTERIOR
            
            img=cv.imread("imagenes/"+fichero)
            gris=cv.cvtColor(cv.resize(img, (64,64)), cv.COLOR_BGR2GRAY)
            fila=gris.reshape(1,-1)
            a=np.append(fila, fichero[0:1]).reshape(1,-1)
            writer.writerows(a)
            
dataframe=pd.read_csv('dataset.csv',header=None)
X=np.array(dataframe.drop([4096],axis=1))
y=dataframe[4096]
for index,(image, label) in enumerate(zip(X[0:6],y[0:6])):
    plt.subplot(2,3,index+1)
    plt.imshow(np.reshape(image,(64,64)),cmap=plt.cm.gray)
    plt.title('%s\n'%str(label), fontsize=18)
#________________________________________________________________________
from sklearn import linear_model
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=4)
model = linear_model.LogisticRegression()
model.fit(X_train,y_train)

predicciones = model.predict(X_test)
print(predicciones)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predicciones))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predicciones))

#PERSISTENCIA DEL MODELO - EPORTAR EL MODELO ENTRENADO
from sklearn.externals import joblib
#import joblib
joblib.dump(model,'modelo_entrenado.pkl') 






  

            
            
            
            
    





