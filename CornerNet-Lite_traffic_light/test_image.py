import glob
import os
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 24 )
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
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets
from capsulelayers import DenseCapsule, PrimaryCapsule
# import time

class CapsuleNet(nn.Module):
    """
    A Capsule Network on MNIST.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """
    def __init__(self, input_size, classes, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=9, stride=1, padding=0)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=9, stride=2, padding=0)

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=32*6*6, in_dim_caps=8,
                                      out_num_caps=classes, out_dim_caps=16, routings=routings)

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(16*classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            # index = length.max(dim=1)[1]
            # print('index',length)
            # y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())
            y = length.clone()
            y[y < 0.5] = 0
            y[y >= 0.5] = 1
            # print('shit',length)
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        # print('re', reconstruction.size())
        # print('return', reconstruction.view(-1, *self.input_size).size())
        return length, reconstruction.view(-1, *self.input_size)

def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
    :param x_recon: reconstructed data, size is same as `x`
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()

    L_recon = nn.MSELoss()(x_recon, x)

    return L_margin + lam_recon * L_recon

def label_totensor(string):
    lbl = []
    if 'GreenSR' == string:
        string = 'SR'
    if 'GreenR' == string:
        string = 'R'
    if 'GreenS' == string:
        string = 'S'
    if 'GreenL' == string:
        string = 'L'
    if 'Green' in string:
        string = string[5:len(string)]
        lbl.append(1)
    else:
        lbl.append(0)
    if 'Yellow' in string:
        string = string[6:len(string)]
        lbl.append(1)
    else:
        lbl.append(0)
    if 'Red' in string:
        string = string[3:len(string)]
        lbl.append(1)
    else:
        lbl.append(0)
    if 'L' in string:
        lbl.append(1)
    else:
        lbl.append(0)
    if 'S' in string:
        lbl.append(1)
    else:
        lbl.append(0)
    if 'R' in string:
        lbl.append(1)
    else:
        lbl.append(0)
    lbl_tensor = torch.FloatTensor(lbl)
    lbl_tensor = lbl_tensor.unsqueeze(0)
    return lbl_tensor

def make_square(im, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    new_im = new_im.resize((28, 28))
    return new_im
# def make_square(img, padColor=0):
#
#     h, w = img.shape[:2]
#     size = 28
#     # size = max(h,w)
#     sh = size
#     sw = size
#
#     # interpolation method
#     if h > sh or w > sw: # shrinking image
#         interp = cv2.INTER_AREA
#     else: # stretching image
#         interp = cv2.INTER_CUBIC
#
#     # aspect ratio of image
#     aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h
#
#     # compute scaling and pad sizing
#     if aspect > 1: # horizontal image
#         new_w = sw
#         new_h = np.round(new_w/aspect).astype(int)
#         pad_vert = (sh-new_h)/2
#         pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
#         pad_left, pad_right = 0, 0
#     elif aspect < 1: # vertical image
#         new_h = sh
#         new_w = np.round(new_h*aspect).astype(int)
#         pad_horz = (sw-new_w)/2
#         pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
#         pad_top, pad_bot = 0, 0
#     else: # square image
#         new_h, new_w = sh, sw
#         pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
#
#     # set pad color
#     if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
#         padColor = [padColor]*3
#
#     # scale and pad
#     scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
#     scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)
#
#     return scaled_img

if __name__ == "__main__":
    import argparse

    model = CapsuleNet(input_size=[3, 28, 28], classes=6, routings=3)
    model.cuda()
    model.load_state_dict(torch.load('/home/bruce/Desktop/CapsNet-Pytorch-master/CapsNet-Pytorch/result/epoch8.pkl'))
    model.eval()
    test_loss = 0
    imgs = []
    lbls = []
    square = []
    tmp1 = []
    tmp2 = []
    tmp3 = []
    tmp4 = []
    count = 0
    number = 0
    # text_green = 'Green'
    # text_yellow = 'Yellow'
    # text_red = 'Red'
    # text_s = 'S'
    # text_l = 'L'
    # text_r = 'R'
    correct = 0
    class_check = np.zeros(6)
    real_right_number = np.zeros(6)
    guess_one_number = np.zeros(6)
    guess_right_number = np.zeros(6)
    image_path = '/media/bruce/54faf521-6795-4648-a136-71ac33061aa3/eddie/acer/data'
    label_path = '/media/bruce/54faf521-6795-4648-a136-71ac33061aa3/eddie/acer/annotation'

    for file in os.listdir(os.path.join(label_path, 'VOLVO_GUO')):
        filepath = os.path.join(label_path, 'VOLVO_GUO', file)
        # print('file', filepath)
        with open(filepath) as fp:
            total = 0
            y_store = []
            x_store = []
            contents = fp.read()
            soup = BeautifulSoup(contents, "html.parser")
            name_find = soup.find_all('name')
            x_min = soup.find_all('xmin')
            for xmin in x_min:
                tmp1.append(xmin.string)
            y_min = soup.find_all('ymin')
            for ymin in y_min:
                tmp2.append(ymin.string)
            x_max = soup.find_all('xmax')
            for xmax in x_max:
                tmp3.append(xmax.string)
            y_max = soup.find_all('ymax')
            for ymax in y_max:
                tmp4.append(ymax.string)
            for name in name_find:
                if name.string[0:6] != 'Square':
                    total += 1
                    lbls.append(name.string)
                    imgs.append(os.path.join(image_path, 'VOLVO_GUO', soup.find('filename').string))
                    square.append((int(tmp1[number]), int(tmp2[number]), int(tmp3[number]), int(tmp4[number])))
                    # print(name.string)

                    # img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                    im = Image.open(imgs[count]).crop(square[count])
                    im = make_square(im)
                    # im = img[int(tmp2[count]):int(tmp4[count]), int(tmp1[count]):int(tmp3[count])]
                    # img = make_square(im)
                    # print(lbls[count], square[count])
                    trans = transforms.ToTensor()
                    im = trans(im)
                    # im = torch.FloatTensor(im)
                    # im = make_square(im)

                    # im_tensor = torch.FloatTensor(im)
                    # print('im', im_tensor.size())

                    # im_tensor = im_tensor.permute(2,0,1)
                    im_tensor = im.unsqueeze(0)
                    lbl_tensor = label_totensor(lbls[count])
                    x, y = Variable(im_tensor.cuda(), volatile=True), Variable(lbl_tensor.cuda())
                    y_pred, x_recon = model(x)
                    # test_loss += caps_loss(y, y_pred, x, x_recon, 0.0005 * 784).data * x.size(0)
                    y_pred[y_pred >= 0.65] = 1
                    y_pred[y_pred < 0.65] = 0
                    y_store.append(y_pred)
                    x_store.append(x_recon)
                    # print('y', y_pred)
                    # print('x',x.size(0))
                    for i in range(x.size(0)):
                        check = True
                        for s in range(len(y_pred[0])):
                            if y_pred[i][s] == y[i][s] and y_pred[i][s] == 1:
                                guess_right_number[s] += 1
                            if y[i][s] == 1:
                                real_right_number[s] += 1
                            if y_pred[i][s] == 1:
                                guess_one_number[s] += 1
                            if y_pred[i][s] == y[i][s]:
                                class_check[s] += 1
                            if y_pred[i][s] != y[i][s]:
                                check = False
                                # print('It is wrong!, It is : ', lbls[count])
                                # print('your guess: ', s)
                                # time.sleep(5)
                        if check == True:
                            correct += 1
                        # if name.string[0:6] != 'Square':
                    count += 1
                    print('correct: ', correct, 'total: ', count, 'acc: ', float(correct)/count)
                number += 1



            img = cv2.imread(imgs[count-1])
            for i in range(total):
                text_string = ''
                if y_store[i][0][0] == 1:
                    text_string += 'Green'
                if y_store[i][0][1] == 1:
                    text_string += 'Yellow'
                if y_store[i][0][2] == 1:
                    text_string += 'Red'
                if y_store[i][0][3] == 1:
                    text_string += 'L'
                if y_store[i][0][4] == 1:
                    text_string += 'S'
                if y_store[i][0][5] == 1:
                    text_string += 'R'
                cv2.rectangle(img, (int(tmp1[number-total+i]), int(tmp2[number-total+i])-40), (int(tmp1[number-total+i])+130, int(tmp2[number-total+i])), (255, 148, 122), -1)
                cv2.putText(img, text_string, (int(tmp1[number-total+i]), int(tmp2[number-total+i])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)





            # print('im', im_tensor.size(), 'y', lbl_tensor)

            cv2.imshow('test_data', img)
            cv2.waitKey(1000)

