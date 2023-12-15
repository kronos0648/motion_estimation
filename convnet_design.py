"""
Copyright (c) 2017, Gavin Weiguang Ding
All rights reserved.

Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Patch

NumDots = 4
NumConvMax = 8
NumFcMax = 20
White = 1.
Bright = 0.9
Light = 0.7
Medium = 0.5
Dark = 0.3
Darker = 0.15
Black = 0.
scale=2

fc_patches=[]
fc_patches_realm=[]


def add_layer(patches, colors, size=(24, 24), num=5,
              top_left=[0, 0],
              loc_diff=[3, -3],
              ):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    for ind in range(num):
        patches.append(Rectangle(loc_start + ind * loc_diff, size[1], size[0]))
        if ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)


def add_layer_with_omission(patches, colors, size=(24, 24),
                            num=5, num_max=8,
                            num_dots=4,
                            top_left=[0, 0],
                            loc_diff=[3, -3],
                            fc=False):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    this_num = min(num, num_max)
    start_omit = (this_num - num_dots) // 2
    end_omit = this_num - start_omit
    start_omit -= 1
    for ind in range(this_num):
        if (num > num_max) and (start_omit < ind < end_omit):
            omit = True
        else:
            omit = False

        if omit:
            patches.append(
                Circle(loc_start + ind * loc_diff + np.array(size)*scale/2, 0.5))
        else:
            rec=Rectangle(loc_start + ind * loc_diff,
                                     size[1]*scale, size[0]*scale)
            
            patches.append(rec)
            if fc==True:
                fc_patches.append(rec)
            

        if omit:
            colors.append(Black)
        elif ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)


def add_mapping(patches, colors, start_ratio, end_ratio, patch_size, ind_bgn,
                top_left_list, loc_diff_list, num_show_list, size_list):

    start_loc = top_left_list[ind_bgn] \
        + (num_show_list[ind_bgn] - 1) * np.array(loc_diff_list[ind_bgn]) \
        + np.array([start_ratio[0] * (size_list[ind_bgn][1] - patch_size[1]),
                    - start_ratio[1] * (size_list[ind_bgn][0] - patch_size[0])]
                   )




    end_loc = top_left_list[ind_bgn + 1] \
        + (num_show_list[ind_bgn + 1] - 1) * np.array(
            loc_diff_list[ind_bgn + 1]) \
        + np.array([end_ratio[0] * size_list[ind_bgn + 1][1],
                    - end_ratio[1] * size_list[ind_bgn + 1][0]])


    patches.append(Rectangle(start_loc, patch_size[1]*scale, -patch_size[0]*scale))
    colors.append(Dark)
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1], end_loc[1]]))
    colors.append(Darker)
    patches.append(Line2D([start_loc[0] + patch_size[1]*scale, end_loc[0]],
                          [start_loc[1], end_loc[1]]))
    colors.append(Darker)
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1] - patch_size[0]*scale, end_loc[1]]))
    colors.append(Darker)
    patches.append(Line2D([start_loc[0] + patch_size[1]*scale, end_loc[0]],
                          [start_loc[1] - patch_size[0]*scale, end_loc[1]]))
    colors.append(Darker)


def label(xy, text, xy_off=[0, 4]):
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
             family='sans-serif', size=20)


def add_neural_line(patches, colors):
    fc_patches_realm.append(len(fc_patches))
    for i in range(len(fc_patches_realm)-2):
        for pA in fc_patches[fc_patches_realm[i]:fc_patches_realm[i+1]]:
            for pB in fc_patches[fc_patches_realm[i+1]:fc_patches_realm[i+2]]:
                xy_a=[pA.xy[0]+pA.get_width(),pA.xy[1]+pA.get_height()/2]
                xy_b=[pB.xy[0],pB.xy[1]+pB.get_height()/2]
                patches.append(ConnectionPatch(xyA=xy_a,xyB=xy_b,coordsA='data',coordsB='data'))
                colors.append(Bright)


def visualize_architecture(fm_sizes:list,fm_depths:list,fm_kernel_sizes:list,fm_stride_sizes:list,fm_texts:list,
                           fc_units:list,fc_texts:list,part:str):
    """Arguments
    
    fm_sizes(featuremap sizes) => [(16, 3), (8, 2), (6, 1), (2, 1), (2, 1)]
    
    fm_depths(featuremap depths) => [3, 16, 6, 6, 6]
    
    fm_kernel_sizes(featuremap kernel sizes) => [(2, 2), (3, 2), (3, 1), None]
    
    fm_stride_sizes(featuremap stride sizes) => [(2,1),(1,2),None,None]
    
    fm_texts(featuremap texts) => ['Convolution', 'Convolution', 'Max-pooling', 'Dropout']
    
    fc_units(fully-connected units) => [12, 96, 96, 3]
    
    fc_texts(fully-connected texts) => ['Flatten\n', 'Fully\nconnected', 'Dropout','Fully\nconnected']
    
    """
    fc_unit_size = 2
    layer_width = 50
    flag_omit = True

    patches = []
    colors = []

    fig, ax = plt.subplots()


    ############################
    # conv layers
    size_list = fm_sizes
    num_list = fm_depths
    x_diff_list = [0, layer_width, layer_width, layer_width]
    text_list = ['Inputs'] + ['Feature\nmaps'] * (len(size_list) - 1)
    loc_diff_list = [[3, -3]] * len(size_list)

    num_show_list = list(map(min, num_list, [NumConvMax] * len(num_list)))
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]

    for ind in range(len(size_list)-1,-1,-1):
        if flag_omit:
            add_layer_with_omission(patches, colors, size=size_list[ind],
                                    num=num_list[ind],
                                    num_max=NumConvMax,
                                    num_dots=NumDots,
                                    top_left=top_left_list[ind],
                                    loc_diff=loc_diff_list[ind])
        else:
            add_layer(patches, colors, size=size_list[ind],
                      num=num_show_list[ind],
                      top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}@{}x{}'.format(
            num_list[ind], size_list[ind][0], size_list[ind][1]),xy_off=[-10, 20])

    ############################
    # in between layers
    start_ratio_list = [[0.0, 0.0]] * len(fm_kernel_sizes)
    end_ratio_list = [[0.0, 0.0]] * len(fm_kernel_sizes)
    patch_size_list = fm_kernel_sizes
    stride_size_list=fm_stride_sizes
    ind_bgn_list = range(len(patch_size_list))
    text_list = fm_texts
    xy_off_bt_layer=[40,-65]
    for ind in range(len(patch_size_list)):
        add_mapping(
            patches, colors, start_ratio_list[ind], end_ratio_list[ind],
            patch_size_list[ind], ind,
            top_left_list, loc_diff_list, num_show_list, size_list)
        if(text_list[ind]=='Dropout'):
            label(top_left_list[ind], text_list[ind], xy_off=xy_off_bt_layer)
            
        elif(text_list[ind]=='Convolution'):
            label(top_left_list[ind], text_list[ind] + '\n{}x{} kernel\nstride=({}x{})'.format(
                patch_size_list[ind][0], patch_size_list[ind][1],stride_size_list[ind][0],stride_size_list[ind][1]), xy_off=xy_off_bt_layer)
        else:
            label(top_left_list[ind], text_list[ind] + '\n{}x{} kernel'.format(
                patch_size_list[ind][0], patch_size_list[ind][1]), xy_off=xy_off_bt_layer)


    ############################
    # fully connected layers
    size_list = [(fc_unit_size, fc_unit_size)] * 4
    num_list = fc_units
    num_show_list = list(map(min, num_list, [NumFcMax] * len(num_list)))
    x_diff_list = [sum(x_diff_list) + layer_width, layer_width, layer_width,layer_width]
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]
    loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
    text_list = ['Hidden\nunits'] * (len(size_list) - 1) + ['Outputs']

    for ind in range(len(size_list)):
        if flag_omit:
            fc_patches_realm.append(len(fc_patches))
            add_layer_with_omission(patches, colors, size=size_list[ind],
                                    num=num_list[ind],
                                    num_max=NumFcMax,
                                    num_dots=NumDots,
                                    top_left=top_left_list[ind],
                                    loc_diff=loc_diff_list[ind],
                                    fc=True)
        else:
            add_layer(patches, colors, size=size_list[ind],
                      num=num_show_list[ind],
                      top_left=top_left_list[ind],
                      loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}'.format(
            num_list[ind]),xy_off=[0, 20])

    text_list = fc_texts

    for ind in range(len(size_list)):
        label(top_left_list[ind], text_list[ind], xy_off=[-10, -65])
    
    add_neural_line(patches, colors)
    

    ############################
    for patch, color in zip(patches, colors):
        patch.set_color(color * np.ones(3))
        if isinstance(patch, Line2D):
            ax.add_line(patch)
        else:
            patch.set_edgecolor(Black * np.ones(3))
            ax.add_patch(patch)

    ax.axis('equal')
    ax.axis('off')
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(sum(x_diff_list))/(r-l)/20
    figh = float(fc_unit_size*NumFcMax)/(t-b)/10
    ax.figure.set_size_inches(figw, figh)
    fig.suptitle(part+'-Data Training\nCNN Model Architecture',y=1.2)
    fig.tight_layout()
    fig_dir = './fig/'
    fig_ext = '.png'
    from datetime import datetime
    fig.savefig(os.path.join(fig_dir, part+'_convnet_'+datetime.now().strftime('%Y%m%d') + fig_ext),
                bbox_inches='tight', pad_inches=0)