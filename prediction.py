#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:04:06 2018

@author: jiabin CHEN
"""

#from keras.preprocessing import image
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
import os
from keras.models import model_from_json
from SudokuPyCSF.mrv_backtracking import search
#import sys

#sys.path.append('/Users/jiabin/INF473I/sodoku_solver/NeuralNetwork-v2/SudokuPyCSF')
parser = argparse.ArgumentParser(
    description='Run a sodoku solver on test images..')

parser.add_argument(
    '-t',
    '--test_path',
    help='path to directory of test images, defaults to Test_images/',
    default='Test_images/sudoku2.jpg')

parser.add_argument(
    '-o',
    '--output_path',
    help='path to output test images, defaults to Result_images/',
    default="Result_images/sudoku2.jpg")


# resize img to 28*28
def resize(rawimg):  
    fx = 28.0 / rawimg.shape[0]
    fy = 28.0 / rawimg.shape[1]
    fx = fy = min(fx, fy)
    img = cv2.resize(rawimg, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    outimg = np.ones((28, 28), dtype=np.uint8) * 255
    w = img.shape[1]
    h = img.shape[0]
    x = (28 - w) // 2
    y = (28 - h) // 2
    outimg[y:y+h, x:x+w] = img
    return outimg



def _main(args):
    
    test_path = os.path.expanduser(args.test_path)
    output_path = os.path.expanduser(args.output_path)

    # load model
    with open('model.json', 'r') as f:        
        model = model_from_json(f.read())
    model.load_weights('model.h5')
    #model.summary()
    
    img = cv2.imread(test_path)
    img_copy = img.copy()
    gray = cv2.cvtColor(img_copy,cv2.COLOR_BGR2GRAY)
    ## Adaptive Threshold 
    #thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,25,25)

    ret,thresh = cv2.threshold(gray,200,255,1)
    
    #Dilate
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5, 5))     
    #dilated = cv2.dilate(thresh,kernel)
 
    #Extract contours
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    ##　choose 81 small cases
    boxes = []
    for i in range(len(hierarchy[0])):
      if hierarchy[0][i][3] == 0:
        boxes.append(hierarchy[0][i])
        
    # external contour
    X, Y, width, height = cv2.boundingRect(contours[0])
    # height, width of small case
    box_h = height/9
    box_w = width/9
    
    # a list for store the description[x,y,w,h] of each case who contain a number
    number_boxes = []
    ## 2D np.array for initialization of soduko
    soduko = np.zeros((9, 9), np.int32)
    #postion for all of cases who contains already number
    pos = []

    for j in range(len(boxes)):
      if boxes[j][2] != -1: # this condition tell us that case box[j] contain a number
        
        x,y,w,h = cv2.boundingRect(contours[boxes[j][2]])
        number_boxes.append([x,y,w,h])
        img = cv2.rectangle(img,(x-1,y-1),(x+w+1,y+h+1),(0,0,255),1)
        img = cv2.drawContours(img, contours, boxes[j][2], (0,255,0), 1)
        
        ## extract this number's region
        number_roi = gray[y:y+h, x:x+w]
 
        ## normalize pixels
        #normalized_roi = number_roi / 255.0  
        res = resize(number_roi)
        res = np.resize(res, (1,28,28,1))
        
        predictions = model.predict(res)
        number = np.argmax(predictions)
        
        ## put result in original image
        
        cv2.putText(img,str(number),(x+w+1,y+h-5), 3, min(width, height)/600.0 , (255, 0, 0), 1, cv2.LINE_AA)
        
        ## Determine the position(x,y) in matrix soduko
        a = int(y/box_h)
        b = int(x/box_w)
        pos.append([a,b])
        soduko[a][b] = number
    
    plt.figure("Detection and Recognition")
    plt.imshow(img)
    
    print("\nAvant resoudre\n")
    print(soduko)
    print("\n")
    # resolve soduko matrix 
    soduko=search(soduko)
    if soduko is None:#this soduko does not have any solution
      print('\nEOF')
      
    else:
      print("\nApres resoudre\n")
      print(soduko)
      print("\nVerification de la somme de chaque colone et ligne\n")
      row_sum = map(sum,soduko)
      col_sum = map(sum,zip(*soduko))
      print(list(row_sum))
      print(list(col_sum))
      
      ## put the result in original image  
      for i in range(9):
        for j in range(9):
          if ([j,i] not in pos):# this number is obtained by algorithme
            #calculate position(x,y) in original image
            x = int((i+0.6)*box_w)
            y = int((j+0.8)*box_h)
            if min(img.shape[0], img.shape[1])>300:
                cv2.putText(img_copy,str(soduko[j][i]),(x,y), 3, min(width, height)/500.0 , (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(img_copy,str(soduko[j][i]),(x,y), 3, min(width, height)/500.0 , (0, 0, 255), 1, cv2.LINE_AA)


       #cv2.imwrite(output_path, img_copy)
      plt.imsave(output_path, img_copy)
      plt.figure("Result")
      plt.imshow(img_copy)
    
    plt.show()
    
    
    
if __name__ == '__main__':
    _main(parser.parse_args())