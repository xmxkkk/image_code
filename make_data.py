import numpy as np
import pandas as pd
import cv2
import os



str = '2345678ABCDEFGHJKLMNPQRTUVWXYABCDEFHIJKMNPQRSTUVWXYZ'
letter_size = len(set(str))
str = sorted(set(str))
str = ''.join(str)

def letter_onehot(lt):
    data = [0] * letter_size
    data[str.index(lt)] = 1
    return np.array(data)


def make_data(files,filename,path):
    datas = []
    labels = []
    idx = 0
    for file in files:
        img = cv2.imread(path + file)
        img2=img.flatten()
        abc = pd.value_counts(img2)

        for x in abc.index:
            if abc[x] > 10000 or abc[x] < 500:
                abc = abc.drop(x)

        for x in abc.index:
            img[img == x] = 1

        img[img != 1] = 0

        img=img[:,:,0]
        img = img[:, :, np.newaxis]
        datas.append(img)

        label = None
        file = file[:5].upper()
        for lt in file:
            oh = letter_onehot(lt)
            oh = oh[np.newaxis, :]
            if label is None:
                label = oh
            else:
                label = np.append(label, oh, axis=0)

        labels.append(label)

        if idx % 1000 == 999 and idx>0:
            print(idx)

        idx = idx + 1
    np.savez_compressed("./data/split/data{}-{}.npz".format(len(files),filename), X=datas, y=labels)


def maker(batch_size=10000,path=''):
    files = os.listdir(path)
    pageSize = int(len(files) / batch_size) + 1

    for i in range(pageSize):
        start = batch_size * i
        end = batch_size * i + batch_size
        if end > len(files):
            end = len(files)

        make_data(files[start:end], "page{}".format(i),path)

maker(10000,"/Users/xmx/Desktop/app/datasets/5code/train/")
# make(10000,'test',"/Users/xmx/Desktop/app/datasets/5code/test/")


'''
batch_size=10000

path="/Users/xmx/Desktop/app/datasets/5code/train/"
files=os.listdir(path)
path="/Users/xmx/Desktop/app/datasets/5code/test/"
files=files+os.listdir(path)

pageSize=int(len(files)/batch_size)+1
'''

    # print("{}/{}/{}".format(start,end,i))