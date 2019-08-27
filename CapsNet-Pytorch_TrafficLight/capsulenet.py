"""
Pytorch implementation of CapsNet in paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this.

Usage:
       Launch `python CapsNet.py -h` for usage help

Result:
    Validation accuracy > 99.6% after 50 epochs.
    Speed: About 73s/epoch on a single GTX1070 GPU card and 43s/epoch on a GTX1080Ti GPU.

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Pytorch`
"""

import glob
import os
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
import cv2
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets
from capsulelayers import DenseCapsule, PrimaryCapsule
from tensorboardX import SummaryWriter



def make_square(im, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    # new_im = new_im.resize((50, 50), Image.BILINEAR)
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


class dogDataset(Dataset):

    def __init__(self, function, root, transform):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
        # path = '/media/bruce/54faf521-6795-4648-a136-71ac33061aa3/eddie/acer'
        # function_image = 'image'

        self.imgs = []
        self.lbls = []
        self.square = []
        tmp1 = []
        tmp2 = []
        tmp3 = []
        tmp4 = []
        self.transform = transform
        count = 0
        label_path = '/media/bruce/54faf521-6795-4648-a136-71ac33061aa3/eddie'

        print('programming--------------------------------')

        import pandas as pd
        url = '/media/bruce/54faf521-6795-4648-a136-71ac33061aa3/eddie/acer'
        iris_tsv_df = pd.read_csv(os.path.join(url, function), sep = ",")
        for item in iris_tsv_df['path']:
            # if item[len(item)-5] == '0':
            filepath = os.path.join(label_path, (item[0:len(item)-6] + '.xml'))
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
                    # print('name', name.string[0:6])
                    if name.string[0:6] != 'Square':
                        # print('name', name.string[0:6], name.string, (int(tmp1[count]), int(tmp2[count]), int(tmp3[count]), int(tmp4[count])))
                        self.lbls.append(name.string)
                        self.imgs.append(os.path.join(root, (item[0:len(item)-6] + '.jpg')))
                        self.square.append((int(tmp1[count]), int(tmp2[count]), int(tmp3[count]), int(tmp4[count])))
                    count += 1
                if count % 1000 == 0:
                    print('load',count)



        # for file in os.listdir(os.path.join(label_path, 'test_label', function)):
        #     filepath = os.path.join(root, 'test_label', function, file)
        #     # print('file', filepath)
        #     with open(filepath) as fp:
        #         contents = fp.read()
        #         soup = BeautifulSoup(contents, "html.parser")
        #         name_find = soup.find_all('name')
        #         x_min = soup.find_all('xmin')
        #         for xmin in x_min:
        #             tmp1.append(xmin.string)
        #         y_min = soup.find_all('ymin')
        #         for ymin in y_min:
        #             tmp2.append(ymin.string)
        #         x_max = soup.find_all('xmax')
        #         for xmax in x_max:
        #             tmp3.append(xmax.string)
        #         y_max = soup.find_all('ymax')
        #         for ymax in y_max:
        #             tmp4.append(ymax.string)
        #         for name in name_find:
        #             if name.string[0:6] != 'Square':
        #                 self.lbls.append(name.string)
        #                 self.imgs.append(os.path.join(root, 'test_picture', function, soup.find('filename').string))
        #                 self.square.append((int(tmp1[count]), int(tmp2[count]), int(tmp3[count]), int(tmp4[count])))
        #             count += 1
        #             print('load',count)

        # for file in os.listdir(os.path.join(root, function)):
        #     filepath = os.path.join(root, function, file)
        #     # print('file', filepath)
        #     with open(filepath) as fp:
        #         contents = fp.read()
        #         soup = BeautifulSoup(contents, "html.parser")
        #         name_find = soup.find_all('name')
        #         x_min = soup.find_all('xmin')
        #         for xmin in x_min:
        #             tmp1.append(xmin.string)
        #         y_min = soup.find_all('ymin')
        #         for ymin in y_min:
        #             tmp2.append(ymin.string)
        #         x_max = soup.find_all('xmax')
        #         for xmax in x_max:
        #             tmp3.append(xmax.string)
        #         y_max = soup.find_all('ymax')
        #         for ymax in y_max:
        #             tmp4.append(ymax.string)
        #         for name in name_find:
        #             if name.string[0:6] != 'Square':
        #                 self.lbls.append(name.string)
        #                 self.imgs.append(os.path.join(root, function_image, soup.find('filename').string))
        #                 self.square.append((int(tmp1[count]), int(tmp2[count]), int(tmp3[count]), int(tmp4[count])))
        #             count += 1
        #             print('load',count)


        # print(len(self.imgs))
        assert len(self.imgs) == len(self.lbls), 'mismatched length!'
        # print ('Total data in {} split: {}'.format(split, len(self.imgs)))

        # label from 0 to (len-1)
        # self.lbls = self.lbls - 1

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        imgpath = self.imgs[index]
        # print('path', imgpath)
        img = cv2.imread(imgpath)
        # im = img[int(self.square[index][1]):int(self.square[index][3]), int(self.square[index][0]):int(self.square[index][2])]
        im = Image.open(imgpath).crop(self.square[index])
        # im.show()
        img = make_square(im)
        # img = imgg.convert('L')

        lbl = []
        if 'GreenSR' == self.lbls[index]:
            self.lbls[index] = 'SR'
        if 'GreenR' == self.lbls[index]:
            self.lbls[index] = 'R'
        if 'GreenS' == self.lbls[index]:
            self.lbls[index] = 'S'
        if 'GreenL' == self.lbls[index]:
            self.lbls[index] = 'L'
        if 'GreenSLR' == self.lbls[index]:
            self.lbls[index] = 'SLR'
        if 'GreenSL' == self.lbls[index]:
            self.lbls[index] = 'SL'
        if 'GreenLR' == self.lbls[index]:
            self.lbls[index] = 'LR'
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
    #


# create train/val transforms
train_transform = trns.Compose([
                      trns.Resize((28, 28)),
                      # trns.RandomCrop((224, 224)),
                      # trns.RandomHorizontalFlip(),
                      trns.ToTensor(),
                      # trns.Normalize(mean=[0.485, 0.456, 0.406],
                      #                std=[0.229, 0.224, 0.225]),
                  ])
test_transform = trns.Compose([
                    trns.Resize((28, 28)),
                    trns.ToTensor(),
                    # trns.Normalize(mean=[0.485, 0.456, 0.406],
                    #                std=[0.229, 0.224, 0.225]),
                ])

# create train/val datasets
trainset = dogDataset(root='/media/bruce/54faf521-6795-4648-a136-71ac33061aa3/eddie/acer',
                      function = 'data_train_list3.csv',
                      transform=train_transform)
testset = dogDataset(root='/media/bruce/54faf521-6795-4648-a136-71ac33061aa3/eddie/acer',
                    function='data_test_list3.csv',
                    transform=test_transform)
# trainset = dogDataset(root='/media/bruce/54faf521-6795-4648-a136-71ac33061aa3/eddie',
#                       function = 'train_data',
#                       transform=train_transform)
# testset = dogDataset(root='/media/bruce/54faf521-6795-4648-a136-71ac33061aa3/eddie',
#                     function='test_data',
#                     transform=test_transform)

# create train/val loaders
train_loader = DataLoader(dataset=trainset,
                          batch_size=20,
                          shuffle=True,
                          num_workers=1)
test_loader = DataLoader(dataset=testset,
                        batch_size=20,
                        shuffle=False,
                        num_workers=1)


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


def show_reconstruction(model, test_loader, n_images, args):
    import matplotlib.pyplot as plt
    from utils import combine_images
    from PIL import Image
    import numpy as np

    model.eval()
    for x, _ in test_loader:
        x = Variable(x[:min(n_images, x.size(0))].cuda(), volatile=True)
        _, x_recon = model(x)
        data = np.concatenate([x.data.cpu(), x_recon.data.cpu()])
        print(data.shape)
        img = combine_images(np.transpose(data, [0, 2, 3, 1]))
        print((img.shape))
        image = img * 255
        print((image.shape))
        Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
        print()
        print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
        print('-' * 70)
        plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png", ))
        plt.show()
        # print('hello')
        break


def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    class_check = np.zeros(6)
    real_right_number = np.zeros(6)
    guess_one_number = np.zeros(6)
    guess_right_number = np.zeros(6)
    for x, y in test_loader:
        # y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)
        x, y = Variable(x.cuda(), volatile=True), Variable(y.cuda())
        # print(x.size())
        y_pred, x_recon = model(x)
        test_loss += caps_loss(y, y_pred, x, x_recon, args.lam_recon).data * x.size(0)  # sum up batch loss
        y_pred_arr = y_pred.cpu().detach().numpy()
        y_true_arr = y.cpu().detach().numpy()
        y_pred_arr[y_pred_arr >= 0.5] = 1
        y_pred_arr[y_pred_arr < 0.5] = 0

        for i in range(x.size(0)):
            check = True
            for s in range(y_pred_arr[0].size):
                if y_pred_arr[i][s] == y_true_arr[i][s] and y_pred_arr[i][s] == 1:
                    guess_right_number[s] += 1
                if y_true_arr[i][s] == 1:
                    real_right_number[s] += 1
                if y_pred_arr[i][s] == 1:
                    guess_one_number[s] += 1
                if y_pred_arr[i][s] == y_true_arr[i][s]:
                    class_check[s] += 1
                if y_pred_arr[i][s] != y_true_arr[i][s]:
                    check = False
            if check == True:
                correct += 1


    test_loss /= len(test_loader.dataset)
    return (test_loss, int(correct) / len(test_loader.dataset), int(class_check[0]) / len(test_loader.dataset),
    int(class_check[1]) / len(test_loader.dataset), int(class_check[2]) / len(test_loader.dataset) ,int(class_check[3]) / len(test_loader.dataset),
    int(class_check[4]) / len(test_loader.dataset), int(class_check[5]) / len(test_loader.dataset), int(guess_right_number[0]) / int(guess_one_number[0]),
    int(guess_right_number[1]) / int(guess_one_number[1]), int(guess_right_number[2]) / int(guess_one_number[2]), int(guess_right_number[3]) / int(guess_one_number[3]),
    int(guess_right_number[4]) / int(guess_one_number[4]), int(guess_right_number[5]) / int(guess_one_number[5]), int(guess_right_number[0]) / int(real_right_number[0]),
    int(guess_right_number[1]) / int(real_right_number[1]), int(guess_right_number[2]) / int(real_right_number[2]), int(guess_right_number[3]) / int(real_right_number[3]),
    int(guess_right_number[4]) / int(real_right_number[4]), int(guess_right_number[5]) / int(real_right_number[5]))


def train(model, train_loader, test_loader, args):
    # from utils import plot_log
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param train_loader: torch.utils.data.DataLoader for training data
    :param test_loader: torch.utils.data.DataLoader for test data
    :param args: arguments
    :return: The trained model
    """
    print('Begin Training' + '-'*70)
    from time import time
    import csv
    logfile = open(args.save_dir + '/log.csv', 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'val_loss', 'val_acc'])
    logwriter.writeheader()
    # plot_log('log.csv')
    #----------------------
    #----------------------
    t0 = time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    best_val_acc = 0.
    writer = SummaryWriter('tensorboard')
    for epoch in range(args.epochs):
        model.train()  # set to training mode
        lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        ti = time()
        training_loss = 0.0

        for i, (x, y) in enumerate(train_loader):  # batch training
            # y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)  # change to one-hot coding
            x, y = Variable(x.cuda()), Variable(y.cuda())  # convert input data to GPU Variable

            optimizer.zero_grad()  # set gradients of optimizer to zero
            y_pred, x_recon = model(x, y)  # forward
            loss = caps_loss(y, y_pred, x, x_recon, args.lam_recon)  # compute loss
            loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
            training_loss += loss.data * x.size(0)  # record the batch loss
            optimizer.step()  # update the trainable parameters with computed gradients
            writer.add_scalar('data/train_loss', loss, i+epoch*len(train_loader))
            # print('iter', i+epoch*len(train_loader))

        # compute validation loss and acc
        val_loss, val_acc, green, yellow, red, l, s, r , gr1, ye1, re1, l1, s1, r1, gr2, ye2, re2, l2, s2, r2 = test(model, test_loader, args)
        logwriter.writerow(dict(epoch=epoch, loss=training_loss / len(train_loader.dataset),
                                val_loss=val_loss, val_acc=val_acc))
        print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.4f\ngreen=%.4f, %.4f, %.4f\nyellow=%.4f, %.4f, %.4f\nred=%.4f, %.4f, %.4f\nl=%.4f, %.4f, %.4f\ns=%.4f, %.4f,%.4f\nr=%.4f, %.4f, %.4f\ntime=%ds"
              % (epoch, training_loss / len(train_loader.dataset),
                 val_loss, val_acc, green, gr1, gr2, yellow, ye1, ye2, red, re1, re2, l, l1, l2, s, s1, s2, r, r1, r2, time() - ti))
        writer.add_scalar('data/loss', val_loss, epoch)
        writer.add_scalar('data/acc', val_acc, epoch)
        writer.add_scalars('data/good', {'Green_acc': green ,'Green_precision': gr1, 'Green_recall': gr2}, epoch)
        writer.add_scalars('data/good', {'Yellow_acc': yellow ,'Yellow_precision': ye1, 'Yellow_recall': ye2}, epoch)
        writer.add_scalars('data/good', {'Red_acc': red ,'Red_precision': re1, 'Red_recall': re2}, epoch)
        writer.add_scalars('data/good', {'L_acc': l ,'L_precision': l1, 'L_recall': l2}, epoch)
        writer.add_scalars('data/good', {'S_acc': s ,'S_precision': s1, 'S_recall': s2}, epoch)
        writer.add_scalars('data/good', {'R_acc': r ,'R_precision': r1, 'R_recall': r2}, epoch)
        if val_acc > best_val_acc:  # update best validation acc and save model
            best_val_acc = val_acc
            print("best val_acc increased to %.4f" % best_val_acc)
        torch.save(model.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)
    logfile.close()
    torch.save(model.state_dict(), args.save_dir + '/trained_model.pkl')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    print("Total time = %ds" % (time() - t0))
    print('End Training' + '-' * 70)
    return model

if __name__ == "__main__":
    import argparse
    import os

    # 0.0005 * 784


    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.0005 * 784, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")  # num_routing should > 0
    parser.add_argument('--shift_pixels', default=2, type=int,
                        help="Number of pixels to shift at most in each direction.")
    parser.add_argument('--data_dir', default='./data',
                        help="Directory of data. If no data, use \'--download\' flag to download it")
    parser.add_argument('--download', action='store_true',
                        help="Download the required data.")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print('args: ', args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    # train_loader, test_loader = load_mnist(args.data_dir, download=True, batch_size=args.batch_size)

    # define model
    model = CapsuleNet(input_size=[3, 28, 28], classes=6, routings=3)
    model.cuda()
    print('model: ', model)
    # check number---------------------------
    light_train = np.zeros(6)
    light_test = np.zeros(6)
    counter_train = 0
    counter_test = 0
    for label in trainset.lbls:
        if 'GreenSR' == label:
            label = 'SR'
        if 'GreenS' == label:
            label = 'S'
        if 'GreenR' == label:
            label = 'R'
        if 'GreenL' == label:
            label = 'L'
        if 'GreenSLR' == label:
            label = 'SLR'
        if 'GreenLR' == label:
            label = 'LR'
        if 'GreenSL' == label:
            label = 'SL'
        if 'Green' in label:
            label = label[5:len(label)]
            light_train[0] += 1
        if 'Yellow' in label:
            label = label[6:len(label)]
            light_train[1] += 1
        if 'Red' in label:
            label = label[3:len(label)]
            light_train[2] += 1
        if 'L' in label:
            light_train[3] += 1
        if 'S' in label:
            light_train[4] += 1
        if 'R' in label:
            light_train[5] += 1
        counter_train += 1

    for label in testset.lbls:
        if 'GreenSR' == label:
            label = 'SR'
        if 'GreenS' == label:
            label = 'S'
        if 'GreenR' == label:
            label = 'R'
        if 'GreenL' == label:
            label = 'L'
        if 'GreenSLR' == label:
            label = 'SLR'
        if 'GreenLR' == label:
            label = 'LR'
        if 'GreenSL' == label:
            label = 'SL'
        if 'Green' in label:
            label = label[5:len(label)]
            light_test[0] += 1
        if 'Yellow' in label:
            label = label[6:len(label)]
            light_test[1] += 1
        if 'Red' in label:
            label = label[3:len(label)]
            light_test[2] += 1
        if 'L' in label:
            light_test[3] += 1
        if 'S' in label:
            light_test[4] += 1
        if 'R' in label:
            light_test[5] += 1
        counter_test += 1
    print('class_train: ', light_train, 'number_train: ', counter_train)
    print('class_test: ', light_test, 'number_test: ', counter_test)
    #------------------------------------

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_state_dict(torch.load(args.weights))
    if not args.testing:
        train(model, train_loader, test_loader, args)
        show_reconstruction(model, test_loader, 50, args)
    else:  # testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test_loss, test_acc, green , yellow, red , l, s, r, gr1, ye1, re1, l1, s1, r1, gr2, ye2, re2, l2, s2, r2 = test(model=model, test_loader=test_loader, args=args)
        print('WTF', test_acc)
        print('loss', test_loss)
        print('test acc = %.4f, test loss = %.5f' % (test_acc, test_loss))

        # show_reconstruction(model, test_loader, 50, args)
    # print(len(test_loader))
