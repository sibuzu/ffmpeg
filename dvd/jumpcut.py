from PIL import Image
import subprocess
import argparse
import numpy as np
import re
import os
import sys
import glob
import shutil
import datetime

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

use_cuda = True
use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
device, kwargs, torch.__version__, sys.version

# argv
parser = argparse.ArgumentParser(description='Modifies a video file to play at different speeds when there is sound vs. silence.')
parser.add_argument('--input_file', type=str,  help='the video file you want modified')
parser.add_argument('--output_file', type=str, default="", help="the output file. (optional. if not included, it'll just modify the input file name)")
parser.add_argument('--sounded_speed', type=float, default=1.00, help="the speed that sounded (spoken) frames should be played at. Typically 1.")
parser.add_argument('--silent_speed', type=float, default=5.00, help="the speed that silent frames should be played at. 999999 for jumpcutting.")
parser.add_argument('--temp_folder', type=str, default="TEMPX", help="temp workdir")

args = parser.parse_args()

INPUT_FILE = args.input_file
if len(args.output_file) >= 1:
    OUTPUT_FILE = args.output_file
else:
    OUTPUT_FILE = INPUT_FILE[:-4] + "x" + INPUT_FILE[-4:]

if len(args.temp_folder) >= 1:
    TEMP_FOLDER = args.temp_folder


TEMP_INPUT = TEMP_FOLDER + "/INPUT"

TEMP_CC = TEMP_FOLDER + "/CC"

def CreateCCFrame():
    shutil.rmtree(TEMP_INPUT, ignore_errors=True)
    os.makedirs(TEMP_INPUT, exist_ok=True)
    shutil.rmtree(TEMP_CC, ignore_errors=True)
    os.makedirs(TEMP_CC, exist_ok=True)

    command = "ffmpeg -i "+INPUT_FILE+" -vf fps=1 "+TEMP_INPUT+"/frame%06d.jpg -hide_banner"
    subprocess.call(command, shell=True)

    area = (346, 421, 506, 453)
    files = sorted(glob.glob(TEMP_INPUT + "/*.jpg"))
    for fname in files:
        img = Image.open(fname)
        outimg = fname.replace("/INPUT", "/CC")
        cropped_img = img.crop(area)
        cropped_img.save(outimg)
        # print(outimg)

# test cc
def normalize(x):
    x = x.convert("YCbCr")
    im = np.array(x, dtype=np.float)
    a = -0.5
    b = 0.5
    y = np.zeros([im.shape[2], im.shape[0], im.shape[1]], dtype=np.float)
    for i in range(3):
        minimum = np.min(im[:,:,i])
        maximum = np.max(im[:,:,i])
        delta = max(maximum - minimum, 0.01)
        y[i,:,:] = a + ((im[:,:,i] - minimum) * (b - a)) / delta
        # print(minimum, maximum)
    return y

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.50),
        )
        self.mlp = nn.Sequential(
            nn.Linear(64064, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
        self.optimizer = optim.Adam(self.parameters())
        self.loss = None

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64064)
        x = self.mlp(x)
        return x

    def train_data(self, data, target):
        self.optimizer.zero_grad()
        output = self.forward(data)
        self.loss = nn.functional.binary_cross_entropy_with_logits(output, target)
        self.loss.backward()
        self.optimizer.step()


def predictModel(model, test):
    model.eval()
    with torch.no_grad():
        testX = torch.from_numpy(test).float().to(device)
        predict = model(testX).cpu()
    predict = np.where(predict > 0, 1, 0)
    return predict

def classified1(model1, files):
    n = len(files)
    test = np.zeros((n, 3, 32, 160), dtype=np.float)
    for i in range(n):
        fname = files[i]
        test[i, :, :, :] = normalize(Image.open(fname))
        
    y = predictModel(model1, test)
    return y

def predictExistCC(model1, ccfiles):
    n = len(ccfiles)
    result = np.zeros((n,1))
    print(n)

    M = (n + 1023) // 1024
    for i in range(M):
        start = i*1024
        end = n if i == M-1 else (i+1)*1024
        files = ccfiles[start:end]
        result[start:end] = classified1(model1, files)

    return result[:,0].tolist()

def calcCCRange(has_cc):
    n = len(has_cc)
    rng = []

    cc = has_cc[0]
    start = 0
    for i in range(1, n):
        if has_cc[i] != cc:
            if cc > 0:
                rng.append((start, i))
            cc = has_cc[i]
            start = i

    if cc > 0:
        rng.append((start, n))
    return rng

def makeClip(rng):
    bw = "+".join(["between(t,{},{})".format(*x) for x in rng])
    vf = "select='{}',setpts=N/FRAME_RATE/TB".format(bw)
    af = "aselect='{}',asetpts=N/SR/TB".format(bw)
    with open("vf.txt", "w") as f:
        f.write(vf)
    with open("af.txt", "w") as f:
        f.write(af)

    command = "ffmpeg -y -i {} -filter_script:v vf.txt -filter_script:a af.txt {} -hide_banner".format(INPUT_FILE, OUTPUT_FILE)
    print(command)
    subprocess.call(command, shell=True)

def dilation(x):
    y = x[1:] + [0]
    z = [0] + x[:-1]
    a = [1 if sum(x)>0 else 0 for x in zip(x, y, z)]
    return a

# START HERE

CreateCCFrame()

model1 = Net1().to(device)
model1.load_state_dict(torch.load("model.mdl"))

ccfiles = glob.glob(TEMP_CC + "/*.jpg")
has_cc = predictExistCC(model1, ccfiles)
# print(has_cc)
has_cc = dilation(has_cc)
# print(has_cc)

rng = calcCCRange(has_cc)
# print(rng)
makeClip(rng)
