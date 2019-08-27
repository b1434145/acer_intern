import cv2
import glob
import os
import torch
from PIL import Image, ImageDraw, ImageFont
font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 24 )
from torch.utils.data.dataset import Dataset
import numpy as np
from torch.utils.data import DataLoader
from scipy.io import loadmat
from PIL import Image
import torchvision
import torchvision.transforms as trns
# import bcolz
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets
from capsulelayers import DenseCapsule, PrimaryCapsule

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

def make_cv2_square(im):
    desired_size = 28
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im

#### aaaaaa

def draw_bboxes_demo(model, image, bboxes, font_size=0.5, thresh=0.5, colors=None):


    """Draws bounding boxes on an image.

    Args:
        image: An image in OpenCV format
        bboxes: A dictionary representing bounding boxes of different object
            categories, where the keys are the names of the categories and the
            values are the bounding boxes. The bounding boxes of category should be
            stored in a 2D NumPy array, where each row is a bounding box (x1, y1,
            x2, y2, score).
        font_size: (Optional) Font size of the category names.
        thresh: (Optional) Only bounding boxes with scores above the threshold
            will be drawn.
        colors: (Optional) Color of bounding boxes for each category. If it is
            not provided, this function will use random color for each category.

    Returns:
        An image with bounding boxes.
    """

    image = image.copy()
    cat_name = 'traffic light'
    # img = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
   
    keep_inds = bboxes[cat_name][:, -1] > thresh
    cat_size  = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
    if colors is None:
        color = np.random.random((3, )) * 0.6 + 0.4
        color = (color * 255).astype(np.int32).tolist()
    else:
        color = colors[cat_name]

    for bbox in bboxes[cat_name][keep_inds]:
        y_store = []
        x_store = []
        # print(bbox[4],int(bbox[0]/511*1920),int(bbox[1]/511*1080),int(bbox[2]/511*1920),int(bbox[3]/511*1080))
        # square = []
        # tmp1 = []
        # tmp2 = []
        # tmp3 = []
        # tmp4 = []
        # tmp1.append(bbox[0])
        # tmp2.append(bbox[1])
        # tmp3.append(bbox[2])
        # tmp4.append(bbox[3])
        # square.append((tmp1, tmp2, tmp3, tmp4))
        # print(type(img))
        img = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        im = make_cv2_square(img)
        im = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        trans = transforms.ToTensor()
        im = trans(im)
        im_tensor = im.unsqueeze(0)
        # lbl_tensor = label_totensor(lbls[count])
        x = Variable(im_tensor.cuda(), volatile=True)
        
        y_pred, x_recon = model(x)
        y_pred[y_pred >= 0.65] = 1
        y_pred[y_pred < 0.65] = 0
        y_store.append(y_pred)
        x_store.append(x_recon)
    
        bbox = bbox[0:4].astype(np.int32)
        # if bbox[1] - cat_size[1] - 2 < 0:
        #     cv2.rectangle(image,
        #         (bbox[0], bbox[1] + 2),
        #         (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
        #         color, -1
        #     )
        #     cv2.putText(image, cat_name,
        #         (bbox[0], bbox[1] + cat_size[1] + 2),
        #         cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
        #     )
        # else:
        #     cv2.rectangle(image,
        #         (bbox[0], bbox[1] - cat_size[1] - 2),
        #         (bbox[0] + cat_size[0], bbox[1] - 2),
        #         color, -1
        #     )
        #     cv2.putText(image, cat_name,
        #         (bbox[0], bbox[1] - 2),
        #         cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
        #     )
        cv2.rectangle(image,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color, 2
        )
        text_string = ''
        if y_store[0][0][0] == 1:
            text_string += 'Green'
        if y_store[0][0][1] == 1:
            text_string += 'Yellow'
        if y_store[0][0][2] == 1:
            text_string += 'Red'
        if y_store[0][0][3] == 1:
            text_string += 'L'
        if y_store[0][0][4] == 1:
            text_string += 'S'
        if y_store[0][0][5] == 1:
            text_string += 'R'
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])-40), (int(bbox[0])+130, int(bbox[1])), (255, 148, 122), -1)
        cv2.putText(image, text_string, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1, cv2.LINE_AA)
       
    return image
