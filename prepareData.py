#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:12:18 2018

@author: jiabin CHEN

Ce programme consacre a preparer l'ensemble des donnees pour entrainer un reseau neurone simple.
Les donnees viennent le site de University of Surrey, England. Il contient 62 fichiers, 10 fichiers
contiennent respectivement 0-9 chiffres en caracteres d'imprimerie, et 26 fichers contiennent respectivement
a-z lettres en minuscule, et l'autre 26 fichers contiennent respectivement A-Z lettres en majuscule.
Ici, pour ce reseau nerone simple qui s'occupe seulement de connaitre les chiffres, donc nous n'utiliserons que
les fichiers contenant chiffres pour l'entrainement et la verification. 


"""
import requests
from tqdm import tqdm
import os
import tarfile
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


# Telecharger les donnees
fileurl = 'http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz'
filename = 'EnglishFnt.tgz'
if not os.path.exists(filename):
    r = requests.get(fileurl, stream=True)
    with open(filename, 'wb') as f:
        for chunk in tqdm(r.iter_content(1024), unit='KB', total=int(r.headers['Content-Length'])/1024): 
            f.write(chunk)
            


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rmdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        
# extraire ces donnees dans 62 fichers        
filename = 'EnglishFnt.tgz'
with tarfile.open(filename, 'r') as tfile:
    print('loading')
    members = tfile.getmembers()
    for member in tqdm(members):
        if tarfile.TarInfo.isdir(member):
            mkdir(member.name)
            continue
        with open(member.name, 'wb') as f:
            f.write(tfile.extractfile(member).read())

# Comme les lettres ne sont pas utils ici pour reseau nerone, donc on transmettre tous les 52 fichers 
# qui contienent lettres dans un seul ficher et puis remove tous les 51 fichers
# Enfin, maintenant on a seulement 11 fichers, 10 contiennent respectivement les chiffres et l'autre contient
# tous les lettres en miniuscule et majuscule
notnumdir = 'English/Fnt/Sample011/'
for i in tqdm(range(12, 63)):
    path = 'English/Fnt/Sample%03d/' % i
    for filename in os.listdir(path):
        os.rename(path+filename, notnumdir+filename)
    os.rmdir(path)
    
    

# resize img to 28*28
def resize(rawimg):  
    fx = 28.0 / rawimg.shape[0]
    fy = 28.0 / rawimg.shape[1]
    fx = fy = min(fx, fy)
    img = cv2.resize(rawimg, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    outimg = np.ones((28, 28), dtype=np.uint8) * 255
    w = img.shape[1]
    h = img.shape[0]
    x = (28 - w) // 2 #division floor
    y = (28 - h) // 2
    outimg[y:y+h, x:x+w] = img
    return outimg
# extraire le centre d'image contenant le chiffre 
def convert(imgpath):
    #img = cv2.imread(imgpath)
    gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 25)
    img2, ctrs, hier = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    x, y, w, h = rects[-1]
    roi = gray[y:y+h, x:x+w]
    return resize(roi)

# On cree ici un ficher 'train', tous les images viennent les fichiers precedents et suites une traitement 'convert'
# definit au dessus
rmdir('train')
for i in range(10):
    path = 'English/Fnt/Sample%03d/' % (i+1)
    trainpath = 'train/%d/' % i
    mkdir(trainpath)
    for filename in tqdm(os.listdir(path), desc=trainpath):
        try:
            cv2.imwrite(trainpath + filename, convert(path + filename))
        except:
            pass
        
# Utiliser une methode 'train_test_split' from package 'sklearn' pour diviser les donnees en 'train' et 'valid'
# en proportion 10:1
for i in range(10):
    trainpath = 'train/%d/' % i
    validpath = 'valid/%d/' % i
    mkdir(validpath)
    imgs = os.listdir(trainpath)
    trainimgs, validimgs = train_test_split(imgs, test_size=0.1)
    for filename in validimgs:
        os.rename(trainpath+filename, validpath+filename)