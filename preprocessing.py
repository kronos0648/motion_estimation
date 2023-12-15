#3차원 벡터 데이터를 RGB 데이터로 범위에 맞게 전처리

from dataclasses import dataclass
import numpy as np
import csv
import sys
import pandas as pd
from imager import Imager,images_to_csv,images_to_csv_test
import asyncio
from enum import Enum
import json

colname={'datetime':0,
            'armAccX':1,'armAccY':2,'armAccZ':3,
            'armGyroX':4,'armGyroY':5,'armGyroZ':6,
            'armAngleX':7,'armAngleY':8,'armAngleZ':9,
            'legAccX':10,'legAccY':11,'legAccZ':12,
            'legGyroX':13,'legGyroY':14,'legGyroZ':15,
            'legAngleX':16,'legAngleY':17,'legAngleZ':18}


@dataclass
class IMUData:
    acc:np.array=None
    gyro:np.array=None
    angle:np.array=None
    
@dataclass
class Part:
    arm='arm'
    leg='leg'
    both='both'


class RGBConverter:
    
    #원점으로부터의 진폭 최대 범위 ex) 진폭=90 => -90<값<90
    amplitude={
        'acc' : 1.2,
        'gyro' : 180,
        'angle' : 180
    }
    
    def __init__(self,rawdata : np.array):
        rawdata=rawdata.T
        self.rawDatas=[]
        self.rgbDatas=None
        self.time_lapse=rawdata[0]
        rawArm=IMUData()
        rawArm.acc=rawdata[1:4]
        rawArm.gyro=rawdata[4:7]
        rawArm.angle=rawdata[7:10]
        self.rawDatas.append(rawArm)
        rawLeg=IMUData()
        rawLeg.acc=rawdata[10:13]
        rawLeg.gyro=rawdata[13:16]
        rawLeg.angle=rawdata[16:19]
        self.rawDatas.append(rawLeg)
    
    
    def convert(self):
        self.rgbDatas=[]
        for raw in self.rawDatas:
            rgbTemp=IMUData()
            rgbTemp.acc=raw.acc+RGBConverter.amplitude['acc']
            rgbTemp.gyro=raw.gyro+RGBConverter.amplitude['gyro']
            rgbTemp.angle=raw.angle+RGBConverter.amplitude['angle']
            
            rgbTemp.acc=np.around(
                (rgbTemp.acc*255/RGBConverter.amplitude['acc']/2).astype(float)
            ).astype(int)
            rgbTemp.gyro=np.around(
                (rgbTemp.gyro*255/RGBConverter.amplitude['gyro']/2).astype(float)
            ).astype(int)
            rgbTemp.angle=np.around(
                (rgbTemp.angle*255/RGBConverter.amplitude['angle']/2).astype(float)
            ).astype(int)
            self.rgbDatas.append(rgbTemp)
           
           

    def getPatternDelimiters(self,part):
        similarRange=4 #유사 시점 값을 일치시키기 위한 범위 계수
        extr_armAngle=[[],[],[]]
        extr_legAngle=[[],[],[]]
        
        extrs:list
        if(part==Part.both):
            extrs=[[],[]]
            extrs[0].append(extr_armAngle)
            extrs[1].append(extr_legAngle)
        else:
            extrs=[[]]
            if(part==Part.arm):
                extrs[0].append(extr_armAngle)
            else:
                extrs[0].append(extr_legAngle)

        
        amount=len(self.time_lapse)
        armData:IMUData=self.rgbDatas[0]
        legData:IMUData=self.rgbDatas[1]
        
        for axis in range(0,3):
            for t in range(1,amount-1):
                tilt_a=armData.angle[axis][t]-armData.angle[axis][t-1]
                tilt_b=armData.angle[axis][t+1]-armData.angle[axis][t]
                if(np.sign(tilt_a)!=np.sign(tilt_b)):
                    extr_armAngle[axis].append(t)
                    
                tilt_a=legData.angle[axis][t]-legData.angle[axis][t-1]
                tilt_b=legData.angle[axis][t+1]-legData.angle[axis][t]
                if(np.sign(tilt_a)!=np.sign(tilt_b)):
                    extr_legAngle[axis].append(t)
               
        if(part==Part.both):
            extr_xyz=[[],[]]
        else:
            extr_xyz=[[]]
        
        for part in range(0,len(extrs)):
            for extr in extrs[part]:
                extr_tmp=np.intersect1d(np.intersect1d((np.array(extr[0]).astype(float)/similarRange).astype(int),(np.array(extr[1]).astype(float)/similarRange).astype(int)),(np.array(extr[2]).astype(float)/similarRange).astype(int))*similarRange
                extr_xyz[part].append(extr_tmp)
                
        final_extr=[]
        for _extr_xyz in extr_xyz:
            final_extr.append([_extr_xyz[0]])
        for part in range(0,len(extrs)):
            for extr in extr_xyz[part]:
                final_extr[part]=np.intersect1d((np.array(final_extr[part]).astype(float)/similarRange).astype(int),(np.array(extr).astype(float)/similarRange).astype(int))*similarRange

        if(len(extrs)==2):
            f_extr=np.intersect1d(final_extr[0],final_extr[1])
        else:
            f_extr=final_extr[0]
        print(f_extr)
        return f_extr
        
    
    
    def getMergedRGB(self):
        rgbMerged=self.time_lapse
        for rgb in self.rgbDatas:
            rgbMerged=np.vstack((rgbMerged,rgb.acc))
            rgbMerged=np.vstack((rgbMerged,rgb.gyro))
            rgbMerged=np.vstack((rgbMerged,rgb.angle))
        
        rgbMerged=rgbMerged.T
        return rgbMerged
        
    def getSplitMergedRGB(self,delimeter:list):
        rgbMerged=self.time_lapse
        for rgb in self.rgbDatas:
            rgbMerged=np.vstack((rgbMerged,rgb.acc))
            rgbMerged=np.vstack((rgbMerged,rgb.gyro))
            rgbMerged=np.vstack((rgbMerged,rgb.angle))
        
        rgbMerged=rgbMerged.T
        splitMerged=[]
        for i in range(len(delimeter)-1):
            splitMerged.append(rgbMerged[delimeter[i]:delimeter[i+1]])
            #print(rgbMerged[delimeter[i]:delimeter[i+1]].shape)
        return splitMerged
            
    def writeCSV(self,rgbMerged,filename):
        with open('rgbdata/'+filename+'.csv','w',newline='') as f:
            writer=csv.writer(f)
            writer.writerows(rgbMerged)
        
    

def proceedImageMaking():
    import os
    fn_rawdatas=os.listdir('rawdata')
    fn_motion=[]
    motion_name:list
    motion_part=[]
    motion_img_seq=[]
    with open('motion.json','r') as f:
        motion_json=json.load(f)
        motion_name=list(motion_json.keys())
        for name in motion_name:
            motion_part.append(motion_json[name])
            motion_img_seq.append(0)
    #motion_name=['arm_left','arm_straight','arm_up','run','walk']
    #motion_part=[Part.arm,Part.arm,Part.arm,Part.leg,Part.leg]
    #motion_img_seq=[0,0,0,0,0]
    for i in range(0,len(motion_name)):
        fn_motion.append([])
    for fn in fn_rawdatas:
        for i in range(0,len(motion_name)):
            if(motion_name[i] in fn):
                fn_motion[i].append(fn)
                break
        
    rawdata_motion=[]
    for i in range(0,len(motion_name)):
        rawdata_motion.append([])
        
    for i in range(0,len(motion_name)):
        for fn in fn_motion[i]:
            _rawdata=pd.read_csv('rawdata/'+fn,header=None,index_col=None)
            _rawdata=_rawdata.to_numpy()
            _converter=RGBConverter(_rawdata)
            _converter.convert()
            _delimeter=_converter.getPatternDelimiters(motion_part[i])
            motion_img_seq[i]=Imager.saveImages(imgs=Imager.makeSplitImage(_converter.getSplitMergedRGB(_delimeter)),filename=motion_name[i],img_seq=motion_img_seq[i])

    print('Max Row Length : ',Imager.max_row_length)
    images_to_csv()
    
            
        
def proceedTestImageMaking():
    import os
    fn_rawdatas=os.listdir('test_rawdata')
    fn_motion=[]
    motion_name:list
    motion_part=[]
    motion_img_seq=[]
    with open('motion.json','r') as f:
        motion_json=json.load(f)
        motion_name=list(motion_json.keys())
        for name in motion_name:
            motion_part.append(motion_json[name])
            motion_img_seq.append(0)
    for i in range(0,len(motion_name)):
        fn_motion.append([])
    for fn in fn_rawdatas:
        for i in range(0,len(motion_name)):
            if(motion_name[i] in fn):
                fn_motion[i].append(fn)
                break
        
    
    rawdata_motion=[]
    for i in range(0,len(motion_name)):
        rawdata_motion.append([])
        
        
    for i in range(0,len(motion_name)):
        for fn in fn_motion[i]:
            _rawdata=pd.read_csv('test_rawdata/'+fn,header=None,index_col=None)
            print(_rawdata)
            _rawdata=_rawdata.to_numpy()
            _converter=RGBConverter(_rawdata)
            _converter.convert()
            _delimeter=_converter.getPatternDelimiters(motion_part[i])
            motion_img_seq[i]=Imager.saveImages(imgs=Imager.makeSplitImage(_converter.getSplitMergedRGB(_delimeter)),filename=motion_name[i],img_seq=motion_img_seq[i],test=True)

    print('Max Row Length : ',Imager.max_row_length)
    images_to_csv_test()
    
    
#proceedImageMaking()
proceedTestImageMaking()

#rawdata=pd.read_csv('rawdata/'+sys.argv[1]+'.csv',header=None,index_col=None)
#rawdata=rawdata.to_numpy()
#converter=RGBConverter(rawdata)
#converter.convert()
#converter.writeCSV(converter.getMergedRGB(),sys.argv[1])
#Imager.saveImages(imgs=Imager.makeSplitImage(converter.getSplitMergedRGB(converter.getPatternDelimiters(Part.arm))),filename=sys.argv[1])

#Imager.saveImage(img=Imager.makeImage(converter.getMergedRGB()),filename=sys.argv[1])

