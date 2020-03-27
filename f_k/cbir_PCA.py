from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import csv
import h5py
import cv2
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras import backend as K
import numpy as np
import os
import time

# 最相似的前K张
k = 10
# 所有图像的路径
all_path=[]
# 所有图像的特征
all_feature=[]
# 输入的图像的路径
one_path=""
# 输入的图像的特征
one_feature=[]
# csv文件路径,读取路径
csv_file = "file_path.csv"
# h5文件，读取特征
h5_file = "feature_4600_PCA.h5"
similarity = []
top_k = []
# 存储label 每次搜索之后刷新label中的image、text
label_img = []
label_txt = []
one_label = []
# 阈值
r = 0.85

def one_img_PCA(one_pic_feature):
    path = 'PCA_array.h5'
    name = 'PCA_array'
    # def openfile(path,name):
    f = h5py.File(path, 'r')
    PCA_array= f[name][:]
    PCA_array = np.transpose(PCA_array)
    f.close()
    one_img_PCA_feature =np.dot(one_pic_feature,PCA_array)       # 矩阵相乘

    return one_img_PCA_feature



def one_img_model_50(path,num_class=71,model_weights="resnet50_weights_top.h5"):

    img_data = cv2.imread(path)
    img_data = cv2.resize(img_data, (224, 224))
    img_data = img_data.reshape(-1, 224, 224, 3)

    global Width, Height

    Width = 224
    Height = 224
    input_tensor = Input(shape=(224, 224, 3))
    base_model = ResNet50(input_tensor=input_tensor, include_top=True, weights='new_vgg/resnet50_weights_base.h5')
    # base_model = ResNet50(input_tensor=input_tensor,include_top=False,weights=None)
    get_resnet50_output = K.function([base_model.layers[0].input, K.learning_phase()],
                                     [base_model.layers[-2].output])
    resnet50_train_output = [get_resnet50_output([img_data, 0])[0]]
    resnet50_train_output = np.concatenate(resnet50_train_output, axis=0)

    input_tensor = Input(shape=(1, 1, 2048))
    x = Flatten()(input_tensor)
    x = Dense(1024, activation='relu')(x)


    predictions = Dense(num_class, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=predictions)
    model.load_weights('new_vgg/'+model_weights)
    get_feature = K.function([model.layers[0].input,K.learning_phase()],[model.layers[-2].output])
    feature_train_output = []
    one_feature_train_output = get_feature([resnet50_train_output, 0])[0]
    feature_train_output.append(one_feature_train_output)
    feature_train_output = np.concatenate(feature_train_output, axis=0)
    return [feature_train_output]

# 余弦相似度
def cosine_similarity(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    # num = vector_a * vector_b.T
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim



# 从csv中读取路径、特征,放入all_path、all_feature
def getData():
    global all_path,all_feature
    # print(len(all_path),len(all_feature))
    with open(csv_file) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            for line in row:
                w = "new_vgg/ydata-yshoes-image-content-v2_0"+"/".join(line.replace(".","",1).split("\\"))
                all_path.append(w)

    f = h5py.File(h5_file, 'r')
    all_feature = f["feature_after_train_PCA"][:]

# 计算相似度
def get_similarity():
    global similarity,one_feature
    del similarity[:]
    one_feature = one_img_model_50(one_path,71)
    PCA_feature = one_img_PCA(one_feature[0][0])
    # print("该图的特征： ",one_feature[0][0])
    print("该图的特征： ",PCA_feature)
    for i in range(len(all_feature)):
        real=np.real(all_feature[i])
        value = cosine_similarity(PCA_feature,real)
        similarity.append(value)


# 得到前k张图片
def get_top_k():
    global similarity,all_path,top_k
    # 统计查全率、查准率、F1
    right = 0
    right_type = one_path.split("/")[-2]

    sp = zip(similarity,all_path)
    sp = sorted(sp,key=lambda x:x[0],reverse=True)
    sim, allpath = zip(*sp)

    # print(len(similarity),similarity[0])
    for n in range(100):
        maxx = sim[n]
        type = allpath[n].split("/")[-2]
        print("第%d相似的图像为%s,余弦相似度为: %.4f" %(n+1,type,maxx))
        top_k.append([allpath[n],sim[n]])
        if type == right_type:
            right += 1

    dirPath = "new_vgg/ydata-yshoes-image-content-v2_0/classes_resize/"+one_path.split("/")[-2]
    NumOfType = len(os.listdir(dirPath))
    print("%s类的图像共有%d张"%(right_type,NumOfType))
    print("前100张属于%s类的共有%d张"%(right_type,right))
    recall = right/NumOfType
    pre = right/100
    if recall*pre == 0 :
        F1 = 0
    else :
        F1 = recall*pre/(2*(recall+pre))
    print("-----------------")
    print("查准率 ： ", pre )
    print("查全率 ： ", recall)
    print("F1 ： ", F1)
    print("-----------------")


# 选取图片，显示相似图片
def helloButton():
    global one_path,top_k,label_img,label_txt,similarity
    del top_k[:]

    # 得到输入图像的完整路径
    one_path = filedialog.askopenfilename(title='打开image文件', filetypes=[('*.png', '*.jpg')])
    # 检索开始时间
    start = time.time()
    print("输入的图片： ",one_path)
    # 读取、缩放 选择的图片
    img = Image.open(one_path)
    img = img.resize((192, 144), Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(img)
    Label(root,image=img_tk).grid(row=0,column=2,columnspan=4, rowspan=3)
    if len(one_label)==0:
        l = Label(root, text=one_path.split('/')[-2], bg="red")
        l.grid(row=3, column=2, columnspan=4)
        one_label.append(l)
    else:
        one_label[0].configure(text=one_path.split('/')[-2])
    # 计算相似性
    get_similarity()
    # 得到前k个
    get_top_k()
    imgs=[]
    # 显示前k个相似图片
    for i in range(k):
        print("第%d相似的像为%s,余弦相似度为: %.4f" % (i+1, top_k[i][0], top_k[i][1]))
        img_k = Image.open(top_k[i][0])
        img_k = img_k.resize((192, 144), Image.ANTIALIAS)
        imgs.append(ImageTk.PhotoImage(img_k))

    i=0
    # 一张图片rowspan=3
    for img in imgs:
        type = top_k[i][0].split("/")[-2]

        if len(label_img) != k:
            # 创建图片label
            l1 = Label(root, image=img)
            label_img.append(l1)
            l1.grid(row=(int(i / 5) + 1) * 4 + 1, column=(int(i % 5)) * 4, columnspan=4, rowspan=3)
            # 创建文本label
            l2 = Label(root, text=type, bg="red")
            label_txt.append(l2)
            l2.grid(row=(int(i / 5) + 1) * 4 + 3 + 1, column=(int(i % 5)) * 4,columnspan=4)
        else:
            label_img[i].configure(image=img)
            label_txt[i].configure(text=type)
        i+=1

    for i in range(len(all_path)):
        if "/Users/yzy/PycharmProjects/f_k/"+all_path[i]==one_path:
            print("find in all_path :  ",all_path[i])
            print("find in all featrue : ",all_feature[i])
            value = cosine_similarity(one_feature[0][0], all_feature[i])
            print("自己和自己的相似度 %f \n\n" % (value))

    end = time.time()
    print("本次检索用时%f秒" % (end - start))
    root.mainloop()


if __name__ == "__main__":

    getData()

    root = Tk()
    Label(root,text = '请选择一张图片：',bg="red").grid(row=0,column=0,rowspan=3,padx=15,pady=1)
    Button(root, text='Choose', command=helloButton).grid(row=0,column=1,rowspan=3,padx=10,pady=5)
    root.geometry('1050x600+200+200')
    root.title("CBIR of Shoes")
    root.mainloop()

