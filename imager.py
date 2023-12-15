#RGB 데이터에 대해 이미지화

import numpy as np
import cv2
import os
import pandas as pd
import time
from tqdm import tqdm
from PIL import Image
import random
import json


class Imager:
    
    max_row_length=0
    
    def makeImage(mergedRgb : np.array):
        img=[]
        rgbMatrix=mergedRgb.T[1:len(mergedRgb)].T
        for x in range(len(rgbMatrix)):
            img.append([])
            for y in range(int((len(rgbMatrix[x]))/3)):
                img[x].append([rgbMatrix[x][y*3],rgbMatrix[x][1+y*3],rgbMatrix[x][2+y*3]])
                
        img=np.array(img,dtype=np.uint8)
        print('image shape : ',img.shape)
        return img
        
    
    def makeSplitImage(splitMergedRgb):
        imgs=[]
        for mergedRgb in splitMergedRgb:
            img=[]
            rgbMatrix=mergedRgb.T[1:len(mergedRgb.T)].T
            for x in range(len(rgbMatrix)):
                img.append([])
                for y in range(int((len(rgbMatrix[x]))/3)):
                    img[x].append([rgbMatrix[x][y*3],rgbMatrix[x][1+y*3],rgbMatrix[x][2+y*3]])
                    
            img=np.array(img,dtype=np.uint8)
            imgs.append(img)
        
        return imgs
        
    def saveImage(img,filename):
        filename='image/'+filename+'.png'
        cv2.imwrite(filename,img)
        print('saved Image')
        
        
    #기존 이미지 번호를 받고 새로 적용될 이미지 번호로 업데이트
    def saveImages(imgs,filename,img_seq,test=False):
        root:str
        if test:
            root='test_image'
        else: root='image'
            
        
        if not os.path.exists(root+'/'+filename):
            os.makedirs(root+'/'+filename)
        
        for img in imgs:
            filepath=root+'/'+filename+'/'+filename+'_'+str(img_seq)+'.png'
            if(len(img)>Imager.max_row_length):
                Imager.max_row_length=len(img)
            cv2.imwrite(filepath,img)
            img_seq+=1
            
        return img_seq
    
    
def images_to_csv():
    image_dirs=os.listdir('image')
    
    # A list for column names of csv
    columnNames = list()
    # A column for label
    columnNames.append('label')
    # Other pixels column
    # replace 144 with your image size, here it is 16x9=144
    # iterate and build headers
    for i in range(288):
        pixel = str(i)
        columnNames.append(pixel)

    # Create a Pandas dataframe for storing data
    train_data = pd.DataFrame(columns = columnNames)

    # calculates the total number of images in the dataset initially 0
    num_images = 0

    # iterate through every folder of the dataset
    for image_dir in image_dirs:

        # print messeage
        print("Iterating: dir[" + image_dir + "]")

        # itreate through every image in the folder
        # tqdm shows progress bar
        for file in tqdm(os.listdir('image/'+image_dir)):
            # open image using PIL Image module
            img = Image.open(os.path.join('image/'+image_dir, file))
            # resize to 28x28, replace with your size
            img = img.resize((16, 6), Image.NEAREST)
            # load image  
            img.load()
            # create a numpy array for image pixels
            imgdata = np.asarray(img, dtype="int32")
            
            motion_dict={}
            motion_name:list
            with open('motion.json','r') as f:
                motion_json=json.load(f)
                motion_name=list(motion_json.keys())
                
            for i in range(len(motion_name)):
                motion_dict[motion_name[i]]=i
            #motion_dict={
            #    'arm_left' : 0,
            #    'arm_straight' : 1,
            #    'arm_up' : 2,
            #    'run' : 3,
            #    'walk' : 4
            #}
            
        
            # temporary array to store pixel values
            data = []
            data.append(motion_dict[image_dir])
            for y in range(16):
                for x in range(6):
                    for rgb in range(3):
                        data.append(imgdata[x][y][rgb])

            # add the data row to training data dataframe
            train_data.loc[num_images] = data

            # increment the number of images
            num_images += 1
    

    # write the dataframe to the CSV file
    train_data.to_csv("motion_train.csv", index=False)
    
    
    #make test data
    #test_data=pd.DataFrame(columns=columnNames)
    
    #for i in range(0,1000):
    #    test_data.loc[i]=train_data.iloc[random.randint(0,train_data.shape[0])]
        
    #test_data.to_csv('motion_test.csv',index=False)
    
    
def images_to_csv_test():
    image_dirs=os.listdir('test_image')
    print('test',image_dirs)
    
    # A list for column names of csv
    columnNames = list()
    # A column for label
    columnNames.append('label')
    # Other pixels column
    # replace 144 with your image size, here it is 16x9=144
    # iterate and build headers
    for i in range(288):
        pixel = str(i)
        columnNames.append(pixel)

    # Create a Pandas dataframe for storing data
    test_data = pd.DataFrame(columns = columnNames)

    # calculates the total number of images in the dataset initially 0
    num_images = 0

    # iterate through every folder of the dataset
    for image_dir in image_dirs:

        # print messeage
        print("Iterating: dir[" + image_dir + "]")

        # itreate through every image in the folder
        # tqdm shows progress bar
        for file in tqdm(os.listdir('test_image/'+image_dir)):
            # open image using PIL Image module
            img = Image.open(os.path.join('test_image/'+image_dir, file))
            # resize to 28x28, replace with your size
            img = img.resize((16, 6), Image.NEAREST)
            # load image  
            img.load()
            # create a numpy array for image pixels
            imgdata = np.asarray(img, dtype="int32")
            
            motion_dict={}
            motion_name:list
            with open('motion.json','r') as f:
                motion_json=json.load(f)
                motion_name=list(motion_json.keys())
                
            print(motion_name)
            for i in range(len(motion_name)):
                motion_dict[motion_name[i]]=i
            
            #motion_dict={
            #    'arm_left' : 0,
            #    'arm_straight' : 1,
            #    'arm_up' : 2,
            #    'run' : 3,
            #    'walk' : 4
            #}
            
        
            # temporary array to store pixel values
            data = []
            data.append(motion_dict[image_dir])
            for y in range(16):
                for x in range(6):
                    for rgb in range(3):
                        data.append(imgdata[x][y][rgb])

            # add the data row to training data dataframe
            test_data.loc[num_images] = data

            # increment the number of images
            num_images += 1
    

    # write the dataframe to the CSV file
    test_data.to_csv("motion_test.csv", index=False)
    
    
