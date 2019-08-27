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
# for file in os.listdir(os.path.join(label_path, function)):
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
            index = length.max(dim=1)[1]
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
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

def make_square(im, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    # new_im = new_im.resize((50, 50), Image.BILINEAR)
    return new_im


class dogDataset(Dataset):

    def __init__(self, root, transform):

        self.imgs = []
        self.lbls = []
        self.square = []
        tmp1 = []
        tmp2 = []
        tmp3 = []
        tmp4 = []
        self.transform = transform
        count = 0
        # label_path = '/media/bruce/54faf521-6795-4648-a136-71ac33061aa3/eddie'
        image_path = '/media/bruce/54faf521-6795-4648-a136-71ac33061aa3/eddie/acer/test_data'
        label_path = '/media/bruce/54faf521-6795-4648-a136-71ac33061aa3/eddie/acer/test_data'
        print('hello')

        for file in os.listdir(os.path.join(label_path, 'label')):
            filepath = os.path.join(label_path, 'label', file)
            # print('file', filepath)
            with open(filepath) as fp:
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
                        self.lbls.append(name.string)
                        self.imgs.append(os.path.join(image_path, 'image', soup.find('filename').string))
                        self.square.append((int(tmp1[count]), int(tmp2[count]), int(tmp3[count]), int(tmp4[count])))
                    count += 1
                    print('load',count)

        # with open(filepath) as fp:
        #     contents = fp.read()
        #     soup = BeautifulSoup(contents, "html.parser")
        #     name_find = soup.find_all('name')
        #     x_min = soup.find_all('xmin')
        #     for xmin in x_min:
        #         tmp1.append(xmin.string)
        #     y_min = soup.find_all('ymin')
        #     for ymin in y_min:
        #         tmp2.append(ymin.string)
        #     x_max = soup.find_all('xmax')
        #     for xmax in x_max:
        #         tmp3.append(xmax.string)
        #     y_max = soup.find_all('ymax')
        #     for ymax in y_max:
        #         tmp4.append(ymax.string)
        #     for name in name_find:
        #         if name.string[0:6] != 'Square':
        #             self.lbls.append(name.string)
        #             self.imgs.append(os.path.join(root, soup.find('filename').string))
        #             self.square.append((int(tmp1[count]), int(tmp2[count]), int(tmp3[count]), int(tmp4[count])))
        #         count += 1
        #         print('load',count)
        # print(len(self.imgs))
    # assert len(self.imgs) == len(self.lbls), 'mismatched length!'


    def __getitem__(self, index):

        imgpath = self.imgs[index]
        im = Image.open(imgpath).crop(self.square[index])
        img = make_square(im)
        lbl = []
        if 'GreenSR' == self.lbls[index]:
            self.lbls[index] = 'SR'
        if 'GreenR' == self.lbls[index]:
            self.lbls[index] = 'R'
        if 'GreenS' == self.lbls[index]:
            self.lbls[index] = 'S'
        if 'GreenL' == self.lbls[index]:
            self.lbls[index] = 'L'
        if 'Green' in self.lbls[index]:
            self.lbls[index] = self.lbls[index][5:len(self.lbls[index])]
            lbl.append(1)
        else:
            lbl.append(0)
        if 'Yellow' in self.lbls[index]:
            self.lbls[index] = self.lbls[index][6:len(self.lbls[index])]
            lbl.append(1)
        else:
            lbl.append(0)
        if 'Red' in self.lbls[index]:
            self.lbls[index] = self.lbls[index][3:len(self.lbls[index])]
            lbl.append(1)
        else:
            lbl.append(0)
        if 'L' in self.lbls[index]:
            lbl.append(1)
        else:
            lbl.append(0)
        if 'S' in self.lbls[index]:
            lbl.append(1)
        else:
            lbl.append(0)
        if 'R' in self.lbls[index]:
            lbl.append(1)
        else:
            lbl.append(0)

        if self.transform is not None:
            img = self.transform(img)

        lbl_tensor = torch.FloatTensor(lbl)

        return img, lbl_tensor

    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.imgs)

test_transform = trns.Compose([
                    trns.Resize((28, 28)),
                    trns.ToTensor(),
                    # trns.Normalize(mean=[0.485, 0.456, 0.406],
                    #                std=[0.229, 0.224, 0.225]),
                ])
testset = dogDataset(root='/home/bruce/Desktop',
                    transform=test_transform)


test_loader = DataLoader(dataset=testset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=2)
# model = TheModelClass(*args, **kwargs)
model = CapsuleNet(input_size=[3, 28, 28], classes=6, routings=3)
model.cuda()
model.load_state_dict(torch.load('/home/bruce/Desktop/CapsNet-Pytorch-master/CapsNet-Pytorch/result/epoch8.pkl'))
model.eval()
test_loss = 0
correct = 0
total = 0
class_check = np.zeros(6)
xx = None
for x, y in test_loader:
    # y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)
    print('x', x.size(0))
    x, y = Variable(x.cuda(), volatile=True), Variable(y.cuda())
    y_pred, x_recon = model(x)
    test_loss += caps_loss(y, y_pred, x, x_recon, 0.0005 * 784).data * x.size(0)  # sum up batch loss
    y_pred_arr = y_pred.cpu().detach().numpy()
    y_true_arr = y.cpu().detach().numpy()
    print('y_predict_prob: ', y_pred_arr )
    y_pred_arr[y_pred_arr >= 0.5] = 1
    y_pred_arr[y_pred_arr < 0.5] = 0
    print('y_predict: ', y_pred_arr )
    # print(xx.size())
    # print(y.size())
    # xx = torch.squeeze(x).cpu()
    # new_img_PIL = transforms.ToPILImage()(xx).convert('RGB')
    # # new_img_PIL = test_loader.dataset[0]
    # new_img_PIL.show()

#     for i in range(x.size(0)):
#         total += 1
#         check = True
#         for s in range(y_pred_arr[0].size):
#             if y_pred_arr[i][s] == y_true_arr[i][s]:
#                 class_check[s] += 1
#             else:
#                 check = False
#         if check == True:
#             correct += 1
#
# test_loss /= len(test_loader.dataset)
# print(test_loss, int(correct) / len(test_loader.dataset), int(class_check[0]) / len(test_loader.dataset),
# int(class_check[1]) / len(test_loader.dataset), int(class_check[2]) / len(test_loader.dataset) ,int(class_check[3]) / len(test_loader.dataset),
# int(class_check[4]) / len(test_loader.dataset), int(class_check[5]) / len(test_loader.dataset))
