import os
from PIL import Image
import numpy as np
import random
import captcha_params

np.random.seed(1337)

# load_data.py and captcha_recognition.py we need to define the MAX_CAPTCHA,the CHAR_SET_LEN ,the tol_num,the train_num and the parameters of the model

# the length of the captcha text
MAX_CAPTCHA = captcha_params.get_captcha_size()
# the number of elements in the char set 
CHAR_SET_LEN = captcha_params.get_char_set_len()

CHAR_SET = captcha_params.get_char_set()

Y_LEN = captcha_params.get_y_len()

height = captcha_params.get_height()
width = captcha_params.get_width()


# return the index of the max_num in the array
def get_max(array):
    max_num = max(array)
    for i in range(len(array)):
        if array[i] == max_num:
            return i

def get_text(array):
    text = []
    max_num = max(array)
    for i in range(len(array)):
        text.append(CHAR_SET[array[i]])
    return text

# text to vector.For example, if the char set is 1 to 10,and the MAX_CAPTCHA is 1
# text2vec(1) will return [0,1,0,0,0,0,0,0,0,0]
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        #print(text)
        print(text_len)
        #raise ValueError(MAX_CAPTCHA)
        # the shape of the vector is 1*(MAX_CAPTCHA*CHAR_SET_LEN)
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    #print(text)

    for i, m in enumerate(text):
        #print(m)
        #print(CHAR_SET.index)
        idx = i * CHAR_SET_LEN + CHAR_SET.index(m.upper())
        vector[idx] = 1
    return vector

def char2pos(c):
    k = CHAR_SET.index(c)
    return k


def load_data(tol_num,train_num,folder):
      
    # input,tol_num: the numbers of all samples(train and test)
    # input,train_num: the numbers of training samples
    # output,(X_train,y_train):trainging data
    # ouput,(X_test,y_test):test data
 
    data = np.empty((tol_num, height, width#,3
                    ),dtype="float32")
    label = np.empty((tol_num,Y_LEN),dtype="uint8")
    texts = np.empty([tol_num,1], dtype="<U10")
    
    

    # data dir
    imgs = os.listdir(folder)
    
    for i in range(tol_num):
        # load the images and convert them into gray images
        img = get_image_from_file(folder+'/'+imgs[i])

        arr = np.asarray(img,dtype="float32")
        #print(arr.shape)
        
        captcha_text = imgs[i].split('.')[0]
        #print(captcha_text,captcha_text.shape)
        if len(captcha_text)!=5:
            continue
        data[i,:,:#,:
            ] = arr
        label[i]= text2vec(captcha_text)
        texts[i]= captcha_text

    # the data, shuffled and split between train and test sets
    rr = [i for i in range(tol_num)] 
    random.shuffle(rr)
    X_train = data[rr][:train_num]
    y_train = label[rr][:train_num]
    y_train_text=texts[rr][:train_num]
    
    X_test = data[rr][train_num:]
    y_test = label[rr][train_num:]
    y_test_text=texts[rr][train_num:]
    
    #Normalize data to -0.5 to 0.5
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train =(X_train/255)-0.5
    X_test =(X_test/255)-0.5
    
    
    # Split y on list of 5 arrays (one for each captcha symbol)

    y_train=y_train.reshape(y_train.shape[0],MAX_CAPTCHA,CHAR_SET_LEN)
    y_test=y_test.reshape(y_test.shape[0],MAX_CAPTCHA,CHAR_SET_LEN)

    y_train_lst=[y_train[:,0,] ,
           y_train[:,1,] , 
           y_train[:,2,] ,
           y_train[:,3,] ,
           y_train[:,4,]  
          ]

    y_test_lst=[y_test[:,0,] ,
           y_test[:,1,] , 
           y_test[:,2,] ,
           y_test[:,3,] ,
           y_test[:,4,]  
          ]
    
    return (X_train.reshape(X_train.shape[0], height, width,1) , y_train_lst,y_train_text),(X_test.reshape(X_test.shape[0], height, width,1),y_test_lst,y_test_text)

def get_image_from_file(path_img):
    img = Image.open(path_img)
    return pre_process_image(img)

def load_image(img):
    tol_num = 1
    data = np.empty((tol_num, 1, height, width),dtype="float32")

    img = pre_process_image(img)

    arr = np.asarray(img,dtype="float32")
    data[0,:,:,:] = arr
    return data


def pre_process_image(img):
    img = img.convert('L')
    # Resize it.
    img = img.resize((width, height), Image.BILINEAR)

    return img


def get_x_input_from_file(img):
    with open(fileName, mode='rb') as file: # b is important -> binary
        fileContent = file.read()

    stream = io.BytesIO(r_data)

    img = Image.open(stream)

    X_test = get_x_input_from_image(img)

    return X_test

def get_x_input_from_image(img):
    X_test = load_image(img)

    X_test = X_test.reshape(X_test.shape[0], height, width, 3)

    X_test = X_test.astype('float32')
    X_test /= 255

    return X_test



