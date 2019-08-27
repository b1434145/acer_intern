# import torch
# y_pred_arr = torch.FloatTensor([[1, 0.5, 0.1], [0, 0.58, 0.77]])
# a = y_pred_arr
# y_pred_arr[y_pred_arr < 0.5] = 0
# y_pred_arr[y_pred_arr >= 0.5] = 1
# print('answer', y_pred_arr)
# print(a)
import pandas as pd
import os
import glob
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from torch.utils.data import DataLoader
from scipy.io import loadmat
from PIL import Image
import os
from bs4 import BeautifulSoup
import torchvision
import torchvision.transforms as trns
import bcolz
import xml.etree.cElementTree as ET
import time

url = '/media/bruce/54faf521-6795-4648-a136-71ac33061aa3/eddie/acer'
function = 'data_train_rebuild.csv'
label_path = '/media/bruce/54faf521-6795-4648-a136-71ac33061aa3/eddie'


def read_xml(file_path):
    tree = ET.ElementTree(file=file_path)


    name = None
    xmin = None
    ymin = None
    xmax = None
    ymax = None

    objs = []

    for obj in tree.iterfind('object'):
        for elem in obj.iter('name'):
            name = elem.text
        for elem in obj.iter('bndbox'):
            for child in elem:
                if child.tag == 'xmin':
                    xmin = child.text
                if child.tag == 'ymin':
                    ymin = child.text
                if child.tag == 'xmax':
                    xmax = child.text
                if child.tag == 'ymax':
                    ymax = child.text
        objs.append([name,xmin,ymin,xmax,ymax])
    # print objs
    return objs


count = 0
iris_tsv_df = pd.read_csv(os.path.join(url, function), sep = ",")
for item in iris_tsv_df['path']:
    if item[len(item)-5] == '0':
        filepath = os.path.join(label_path, (item[0:len(item)-6] + '.xml'))
        t1 = time.time()
        read_xml(filepath)
        print('read xml', time.time()-t1)
        # with open(filepath) as fp:

            # contents = fp.read()
            # soup = BeautifulSoup(contents, "html.parser")
            # name_find = soup.find_all('name')
            # x_min = soup.find_all('xmin')
            # for xmin in x_min:
            #     tmp1.append(xmin.string)
            # y_min = soup.find_all('ymin')
            # for ymin in y_min:
            #     tmp2.append(ymin.string)
            # x_max = soup.find_all('xmax')
            # for xmax in x_max:
            #     tmp3.append(xmax.string)
            # y_max = soup.find_all('ymax')
            # for ymax in y_max:
            #     tmp4.append(ymax.string)
            # for name in name_find:
            #     if name.string[0:6] != 'Square':
            #         self.lbls.append(name.string)
            #         self.imgs.append(os.path.join(root, (item[0:len(item)-6] + '.jpg')))
            #         self.square.append((int(tmp1[count]), int(tmp2[count]), int(tmp3[count]), int(tmp4[count])))
        count += 1
        if count % 1000 == 0:
            print('load',count)
