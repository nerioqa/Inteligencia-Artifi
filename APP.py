# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:17:22 2020
@author: NERIO QUISPE ANCCO
"""
from appJar import gui
import cv2 as cv
import numpy as np
import sys
import os
from sklearn import linear_model
from sklearn.externals import joblib
import sklearn.utils._cython_blas

app=gui("RECONOCIMIENTO DE VOCALES              desarrollador: NERIO QUISPE ANCCO","600X400")
model=joblib.load('modelo_entrenado.pkl')
#________________________________________
def eventos(button):
    if button=='Procesar':
        ruta=app.getEntry("f1")
        if(len(ruta)>0):
            app.setImage("imgvocal",ruta)
            img=cv.imread(ruta)
            gris=cv.cvtColor(cv.resize(img,(64,64)),cv.COLOR_BGR2GRAY)
            X=np.array(gris.reshape(1,-1))
            prediccion=model.predict(X)
            app.infoBox("respuesta","LA VOCAL ES "+str(prediccion[0]), parent=None)
#_________________________________________
app.addFileEntry("f1")
app.addImage("imgvocal", "muestras.png")
app.addButtons(['Nuevo','Procesar','Salir'],eventos)
app.setIcon('favicon.ico')

app.go()


