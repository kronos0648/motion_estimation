import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import keras.losses
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential,load_model
from keras.layers import Dense, Conv2D, AveragePooling2D,MaxPooling2D, Flatten,Dropout
from keras import backend as K
import tensorflow as tf
import keras.metrics
import keras.saving
import os
from threading import Thread
import json
import pickle
import sys


from datetime import datetime

from convnet_design import visualize_architecture

motion_kind:list
motion_dict:dict
with open('motion.json','r') as f:
    motion_dict=dict(json.load(f))
    motion_kind=list(motion_dict.keys())
#motion_kind=['arm_left','arm_straight','arm_up','run','walk']
models:dict={}
historys:dict={}
conf_matrixs:dict={}

motion_matched={}
motion_total_data_num={}

idxs_arm=[]
idxs_leg=[]
idxs={'arm' : idxs_arm,
      'leg' : idxs_leg}

key=['arm','leg']

for i in range(len(motion_kind)):
    if motion_dict[motion_kind[i]]=='arm':
        idxs_arm.append(i)
    else:
        idxs_leg.append(i)

test_idx=11

model_path='model/'
history_path='history/'


def read_scale_dataset2():
    train_data = pd.read_csv("motion_train.csv")
    train_data['label'].value_counts().plot.bar(color='cyan')
    Y_train = train_data['label'].values
    
    X_train = train_data.drop('label', axis=1).values
    
    x_train:dict={}
    x_val:dict={}
    x_test:dict={}
    x_train['arm']=X_train.T[0:144].T
    x_train['leg']=X_train.T[144:288].T
    
    y_train:dict={}
    y_val:dict={}
    y_test:dict={}
    
    y_train['arm']=[]
    y_train['leg']=[]
    
    for i in reversed(range(len(Y_train))):
        if Y_train[i] in idxs_arm:
            x_train['leg']=np.delete(x_train['leg'],i,axis=0)
            y_train['arm'].insert(0,Y_train[i])
        else:
            x_train['arm']=np.delete(x_train['arm'],i,axis=0)
            y_train['leg'].insert(0,Y_train[i])
            
    for k in key:
        y_train[k]=np.array(y_train[k])
        if k=='leg':
            y_train[k]=y_train[k]-3
            
    for k in key:
        x_train[k],x_val[k],y_train[k],y_val[k]=train_test_split(x_train[k],y_train[k],test_size=0.2,stratify=y_train[k])
        
    for k in key:
        x_train[k],x_test[k],y_train[k],y_test[k]=train_test_split(x_train[k],y_train[k],test_size=0.2,stratify=y_train[k])
        
        
    for k in key:
        y_train[k]=to_categorical(y_train[k],num_classes=len(idxs[k]))
        y_val[k]=to_categorical(y_val[k],num_classes=len(idxs[k]))
        y_test[k]=to_categorical(y_test[k],num_classes=len(idxs[k]))
        
        x_train[k] = x_train[k].reshape((x_train[k].shape[0], 16, 3, 3))   
        x_train[k] = x_train[k].astype('float32')
        x_train[k] = x_train[k]/255.0 
        
        x_val[k] = x_val[k].reshape((x_val[k].shape[0], 16, 3, 3))   
        x_val[k] = x_val[k].astype('float32')
        x_val[k] = x_val[k]/255.0 
        
        x_test[k] = x_test[k].reshape((x_test[k].shape[0], 16, 3, 3))   
        x_test[k] = x_test[k].astype('float32')
        x_test[k] = x_test[k]/255.0 
        
    return x_train, y_train, x_val, y_val, x_test, y_test

def read_scale_dataset():
    
    train_data = pd.read_csv("motion_train.csv")
    test_data = pd.read_csv("motion_test.csv")
    
    # visualizing class label frequency in the input data
    train_data['label'].value_counts().plot.bar(color='cyan')
    Y_train = train_data['label'].values
    
    X_train = train_data.drop('label', axis=1).values
    
    X_test = test_data.drop('label', axis=1).values
    
    Y_test = test_data['label'].values
    
    x_train:dict={}
    x_val:dict={}
    x_test:dict={}
    x_train['arm']=X_train.T[0:144].T
    x_train['leg']=X_train.T[144:288].T
    x_test['arm']=X_test.T[0:144].T
    x_test['leg']=X_test.T[144:288].T    
    


    y_train:dict={}
    y_val:dict={}
    y_test:dict={}
    

    
    y_train['arm']=[]
    y_train['leg']=[]
    y_test['arm']=[]
    y_test['leg']=[]
    
    #arm, leg에 각각 부적합한 동작 데이터 제거
    for i in reversed(range(len(Y_train))):
        if Y_train[i] in idxs_arm:
            x_train['leg']=np.delete(x_train['leg'],i,axis=0)
            y_train['arm'].insert(0,Y_train[i])
        else:
            x_train['arm']=np.delete(x_train['arm'],i,axis=0)
            y_train['leg'].insert(0,Y_train[i])
            
    for i in reversed(range(len(Y_test))):
        if Y_test[i] in idxs_arm:
            x_test['leg']=np.delete(x_test['leg'],i,axis=0)
            y_test['arm'].insert(0,Y_test[i])
        else:
            x_test['arm']=np.delete(x_test['arm'],i,axis=0)
            y_test['leg'].insert(0,Y_test[i])
            
    for k in key:
        y_train[k]=np.array(y_train[k])
        y_test[k]=np.array(y_test[k])
        if k=='leg':
            y_train[k]=y_train[k]-3
            y_test[k]=y_test[k]-3
    
    for k in key:
        x_train[k],x_val[k],y_train[k],y_val[k]=train_test_split(x_train[k],y_train[k],test_size=0.2,stratify=y_train[k])
    
    for k in key:
        y_train[k]=to_categorical(y_train[k],num_classes=len(idxs[k]))
        y_val[k]=to_categorical(y_val[k],num_classes=len(idxs[k]))
        y_test[k]=to_categorical(y_test[k],num_classes=len(idxs[k]))
        
        x_train[k] = x_train[k].reshape((x_train[k].shape[0], 16, 3, 3))   
        x_train[k] = x_train[k].astype('float32')
        x_train[k] = x_train[k]/255.0 
        
        x_val[k] = x_val[k].reshape((x_val[k].shape[0], 16, 3, 3))   
        x_val[k] = x_val[k].astype('float32')
        x_val[k] = x_val[k]/255.0 
        
        x_test[k] = x_test[k].reshape((x_test[k].shape[0], 16, 3, 3))   
        x_test[k] = x_test[k].astype('float32')
        x_test[k] = x_test[k]/255.0 
    
    
    
    
    
    return x_train, y_train, x_val, y_val, x_test, y_test


def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -K.log(model_out))
        weight = tf.multiply(y_true, K.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed


#input_shape : 16,3,3
def create_model(part,thresholds):
    model = Sequential([
        Conv2D(16, kernel_size=(2,2), kernel_initializer='he_uniform', input_shape=(16,3,3), strides=(2,1), activation='relu'),
        Conv2D(6, kernel_size=(3,2), kernel_initializer='he_uniform', strides=(1,2), activation='relu'),
        MaxPooling2D((3,1)),
        Flatten(),
        Dense(384, kernel_initializer='he_uniform', activation='relu'),
        Dropout(0.2),
        Dense(len(idxs[part]), activation='softmax')
    ])
    
    model.compile(optimizer='nadam', loss=keras.losses.CategoricalFocalCrossentropy(alpha=1), 
                  metrics=['accuracy',
                           keras.metrics.Precision(thresholds=thresholds),
                           keras.metrics.Recall(thresholds=thresholds),
                           keras.metrics.AUC(thresholds=[thresholds]*len(idxs[part])),
                           keras.metrics.F1Score(threshold=thresholds,average='micro',name='micro_avg'),
                           keras.metrics.F1Score(threshold=thresholds,average='macro',name='macro_avg'),
                           keras.metrics.F1Score(threshold=thresholds,average='weighted',name='weighted_avg'),
                           ])
    """
    model.compile(optimizer='nadam', loss=focal_loss(alpha=2), 
                  metrics=['accuracy',
                           keras.metrics.Precision(thresholds=thresholds),
                           keras.metrics.Recall(thresholds=thresholds),
                           keras.metrics.AUC(thresholds=[thresholds]*len(idxs[part])),
                           tfa.metrics.F1Score(threshold=thresholds,num_classes=len(idxs[part]),average='micro',name='micro_avg'),
                           tfa.metrics.F1Score(threshold=thresholds,num_classes=len(idxs[part]),average='macro',name='macro_avg'),
                           tfa.metrics.F1Score(threshold=thresholds,num_classes=len(idxs[part]),average='weighted',name='weighted_avg')
                           ])
    """

    return model



#팔 다리 총 두번 실행하도록
def evaluate_model(X_train, Y_train, X_val, Y_val, X_test, Y_test,thresholds,part,epoch,model=None,):
    # model fitting
    if model==None:
        model = create_model(part,thresholds)
    history = model.fit(X_train, Y_train, epochs=epoch, batch_size=16,validation_data=(X_val,Y_val),verbose=2)
    print('\n\n')
    model.summary()
    print('\n\n')
    model.evaluate(X_test, Y_test)
    
    return model, history


def visualize(hist:dict,model:dict,conf_matrix:dict):
    fig_dir = './fig/'
    fig_ext = '.png'
    print('\n')
    parts=list(hist.keys())
    #learning curve
    learning_curve=plt.figure(num='learning curve')
    learning_curve.suptitle('Classification Learning Curve',y=1.0)
    ax_size_inches=(20,20)

    for idx in range(len(parts)):
        ax=learning_curve.add_subplot(4, 2*len(parts), 1+idx*8)
        ax.set_title('Accuracy({})'.format(parts[idx]))
        ax.plot(hist[parts[idx]].history['accuracy'], color='blue', label='train')
        ax.figure.set_size_inches(ax_size_inches)
        
        ax=learning_curve.add_subplot(4, 2*len(parts), 2+idx*8)
        ax.set_title('Precision({})'.format(parts[idx]))
        if idx>0:
            ax.plot(hist[parts[idx]].history['precision_'+str(idx)], color='green', label='train')
        else:
            ax.plot(hist[parts[idx]].history['precision'], color='green', label='train')
        ax.figure.set_size_inches(ax_size_inches)
        
        ax=learning_curve.add_subplot(4, 2*len(parts), 3+idx*8)
        ax.set_title('Recall({})'.format(parts[idx]))
        if idx>0:
            ax.plot(hist[parts[idx]].history['recall_'+str(idx)], color='cyan', label='train')
        else:
            ax.plot(hist[parts[idx]].history['recall'], color='cyan', label='train')
        ax.figure.set_size_inches(ax_size_inches)
        
        ax=learning_curve.add_subplot(4, 2*len(parts), 4+idx*8)
        ax.set_title('AUC({})'.format(parts[idx]))
        if idx>0:
            ax.plot(hist[parts[idx]].history['auc_'+str(idx)], color='black', label='train')
        else:
            ax.plot(hist[parts[idx]].history['auc'], color='black', label='train')
        ax.figure.set_size_inches(ax_size_inches)
        
        ax=learning_curve.add_subplot(4, 2*len(parts), 5+idx*8)
        ax.set_title('Micro Average({})'.format(parts[idx]))
        ax.plot(hist[parts[idx]].history['micro_avg'], color='olive', label='train')
        ax.figure.set_size_inches(ax_size_inches)
        
        ax=learning_curve.add_subplot(4, 2*len(parts), 6+idx*8)
        ax.set_title('Macro Average({})'.format(parts[idx]))
        ax.plot(hist[parts[idx]].history['macro_avg'], color='purple', label='train')
        ax.figure.set_size_inches(ax_size_inches)
        
        ax=learning_curve.add_subplot(4, 2*len(parts), 7+idx*8)
        ax.set_title('Weighted Average({})'.format(parts[idx]))
        ax.plot(hist[parts[idx]].history['weighted_avg'], color='brown', label='train')
        ax.figure.set_size_inches(ax_size_inches)
        
        ax=learning_curve.add_subplot(4, 2*len(parts), 8+idx*8)
        ax.set_title('Focal Loss({})'.format(parts[idx]))
        ax.plot(hist[parts[idx]].history['loss'], color='red', label='train')
        ax.figure.set_size_inches(ax_size_inches)
    
    learning_curve.subplots_adjust(wspace=0.9,hspace=0.9)
    learning_curve.tight_layout()
    learning_curve.savefig(os.path.join(fig_dir, 'Learning Curve_'+datetime.now().strftime('%Y%m%d') + fig_ext),
                bbox_inches='tight', pad_inches=0)
    
    #motion precision rate
    motion_rate=plt.figure(num='motion precision rate')
    ax=motion_rate.add_subplot(1,1,1)
    motion_list=list(motion_total_data_num.keys())
    motion_matched_arr=np.array(list(motion_matched.values()),dtype=float)
    motion_total_arr=np.array(list(motion_total_data_num.values()),dtype=float)
    motion_rate_value=motion_matched_arr/motion_total_arr
    motion_rate.suptitle('Motion Prediction Rate',y=1.0)
    ax.bar(motion_list,motion_rate_value)
    ax.figure.set_size_inches(10,5)
    motion_rate.tight_layout()

    
    
    motion_rate.savefig(os.path.join(fig_dir, 'Motion_Prediction_Rate_'+datetime.now().strftime('%Y%m%d') + fig_ext),
                bbox_inches='tight', pad_inches=0)
    
    #CNN architecture
    for idx in range(len(parts)):
        fm_sizes=[]
        fm_depths=[]
        fm_kernel_sizes=[]
        fm_stride_sizes=[]
        fm_texts=[]
        fc_units=[]
        fc_texts=[]
        seq_layers=0
        layers=model[parts[idx]].layers
        for layer in layers:
            config=layer.get_config()
            if seq_layers==0:
                if('conv2d' in config['name']):
                    fm_sizes.append((layer.input.shape[1],layer.input.shape[2]))
                    fm_depths.append(layer.input.shape[3])
                    fm_kernel_sizes.append(layer.get_config()['kernel_size'])
                    fm_stride_sizes.append(layer.get_config()['strides'])
                    fm_texts.append('Convolution')
                elif('max_pooling2d' in config['name']):
                    fm_sizes.append((layer.input.shape[1],layer.input.shape[2]))
                    fm_depths.append(layer.input.shape[3])
                    fm_kernel_sizes.append(layer.get_config()['pool_size'])
                    fm_stride_sizes.append(None)
                    fm_texts.append('Max-pooling')
                elif('dropout' in config['name']):
                    fm_sizes.append((layer.input.shape[1],layer.input.shape[2]))
                    fm_depths.append(layer.input.shape[3])
                    fm_kernel_sizes.append((layer.input.shape[1],layer.input.shape[2]))
                    fm_stride_sizes.append(None)
                    fm_texts.append('Dropout')
                elif('flatten' in config['name']):
                    fm_sizes.append((layer.input.shape[1],layer.input.shape[2]))
                    fm_depths.append((layer.input.shape[3]))
                    fc_texts.append('Flatten')
                    seq_layers+=1
            else:
                if('dense' in config['name']):
                    fc_units.append(layer.input.shape[1])
                    fc_texts.append('Fully\nConnected')
                    if(id(layer)==id(layers[-1])):
                        fc_units.append(layer.output.shape[1])
                elif('dropout' in config['name']):
                    fc_units.append(layer.input.shape[1])
                    fc_texts.append('Dropout')
        visualize_architecture(fm_sizes,fm_depths,fm_kernel_sizes,fm_stride_sizes,fm_texts,fc_units,fc_texts,parts[idx])
    
    #Confusion Matrix
    df_cms={}
    for idx in range(len(parts)):
        motion_title:list
        if idx==0:
            motion_title=motion_kind[0:3]
        else:
            motion_title=motion_kind[3:8]
        df_cm=pd.DataFrame(conf_matrix[parts[idx]],
                           index=motion_title,columns=motion_title)
        df_cms[parts[idx]]=df_cm
    
    conf_matrix_image=plt.figure(num='confusion matrix')
    for idx in range(len(parts)):
        conf_ax=conf_matrix_image.add_subplot(len(parts),1,idx+1)
        heatmap=sn.heatmap(df_cms[parts[idx]],vmin=0,vmax=1,annot=True,ax=conf_ax)
        heatmap.set_title(parts[idx])
        heatmap.figure.set_size_inches(7,7)
        
        
    conf_matrix_image.suptitle('Confusion Matrix',y=1.0)
    conf_matrix_image.tight_layout()
    conf_matrix_image.savefig(os.path.join(fig_dir, 'Confusion Matrix_'+datetime.now().strftime('%Y%m%d') + fig_ext),
                bbox_inches='tight', pad_inches=0)
    
    print('Figure Images Created')
    
    
def runLearn(X_train,Y_train,X_val,Y_val,X_test,Y_test,part,model):
    epoch=100
    thresholds=float(1)-float(1)/float(len(idxs[part]))
    model, history = evaluate_model(X_train, Y_train,X_val,Y_val, X_test, Y_test,thresholds,part,epoch,model)
    models[part]=model
    historys[part]=history
    
    


def predict(model,history,X_test,Y_test,part):
    Y_pred = model.predict(X_test)
    result=confusion_matrix(y_true=np.argmax(Y_test,axis=1),y_pred=np.argmax(Y_pred,axis=1),normalize='pred')
    for i in range(len(Y_test)):
        class_idx=np.argmax(Y_test[i])
        if part=='leg':
            class_idx+=3
        if(np.argmax(Y_test[i])==np.argmax(Y_pred[i])):
            motion_matched[motion_kind[class_idx]]+=1
        motion_total_data_num[motion_kind[class_idx]]+=1
    return result
    



x_train, y_train, x_val, y_val, x_test, y_test = read_scale_dataset2()


# Checking authenticity of data
#plt.imshow(X_train_arm[440].reshape(16,3,3))
#plt.imshow(X_train_leg[440].reshape(16,3,3))

f=open('result/result_'+datetime.now().strftime('%Y%m%d')+'.txt','w')
sys.stdout=f

for k in key:
    models[k]=None

for k in key:
    if(len(sys.argv)==2):
        models[k]=load_model('model/model_'+k+'_'+sys.argv[1])
        #models[k]=load_model('model/model_'+k+'_'+sys.argv[1],custom_objects={'focal_loss_fixed' : focal_loss(gamma=2, alpha=1)})
    

learner:dict={}

for k in key:
    learner[k]=Thread(target=runLearn(X_train=x_train[k],Y_train=y_train[k],X_val=x_val[k],Y_val=y_val[k],X_test=x_test[k],Y_test=y_test[k],part=k,model=models[k]))

for k in key:
    learner[k].start()

for k in key:
    learner[k].join()

#just for predict
#models['arm']=model_arm
#models['leg']=model_leg
#historys['arm']=None
#historys['leg']=None

for motion in motion_kind:
    motion_matched[motion]=0
    motion_total_data_num[motion]=0

for k in key:
    conf_matrixs[k]=predict(models[k],historys[k],x_test[k],y_test[k],part=k)

print('Motion Precision Rate(Test Data)')
for motion in motion_kind:
    print(motion,'\t',motion_matched[motion],' / ',motion_total_data_num[motion])

visualize(historys,models,conf_matrixs)
f.close()

for k in key:
    models[k].save(model_path+'model_'+k+'_'+datetime.now().strftime('%Y%m%d'))
    
for k in key:
    with open(history_path+'history_'+k+'_'+datetime.now().strftime('%Y%m%d'),'wb') as hist_file:
        pickle.dump(historys[k],hist_file)
