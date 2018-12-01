import shutil
import os
import numpy as np
import cv2
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

from skimage.transform import rescale, resize, downscale_local_mean

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K





source = "/home/p4bhattachan/PycharmProjects/syde770/images/rgb/"
destination = "/home/p4bhattachan/PycharmProjects/syde770/images/depth/"


def move(source_dir,destination_dir,match_str):
    i = 0
    j = 0

    for f in os.listdir(source):
        j += 1
        if match_str in f:
            filename = source_dir + f
            shutil.move(filename, destination_dir)
            i += 1

    print("total files = {} number of files moved = {}".format(j,i))

def copy(source_dir,destination_dir,num=100):
    j = 0

    for f in os.listdir(source_dir):
        if (j < num):
            filename = source_dir + f
            shutil.copy(filename, destination_dir)
            j += 1

    print("total files copied = {}".format(j))

def image_shape(dirname):
    j=0
    for f in os.listdir(dirname):
        img = np.load(dirname + f)
        if "depth" in f:
            my_tup = (480,640)
        else:
            my_tup = (480,640,3)

        if img.shape != my_tup:
            print("image = {} - non-standard image size {}".format(f, img.shape))
        else:
            j +=1

        plt.imshow("RS", img)  # opens CV window with image overlaid on frames
        plt.show()

#image scaling using skimage and ensuring that rgb and depth exis
def copy_and_rescale_images(rgb_dir, depth_dir, new_size):

    j = 0
    for f in os.listdir(rgb_dir):
        img = np.load(rgb_dir + f)
        # cv2.resize requires a tupe of w x h - crazy
        new_rgb_image = resize(img, (new_size[0], new_size[1]), anti_aliasing=True)
        new_rgb_file = rgb_dir+f[:15]+"_reshaped"+".npy"


        for d in os.listdir(depth_dir):
            if f[:6] in d:
                d_img = np.load(depth_dir + d)
                new_depth_image = resize(d_img,(new_size[0], new_size[1]), anti_aliasing=True)
                new_depth_file = depth_dir + d[:15] + "_reshaped" + ".npy"
                np.save(new_rgb_file, new_rgb_image)
                np.save(new_depth_file, new_depth_image)
                j+=1
                break

        if (j > 99):
            break

def copy_and_rescale_images_for_testing(rgb_dir, depth_dir, new_size):
    j = 0
    des_dir = "/home/p4bhattachan/PycharmProjects/syde770/images/test/"
    for f in os.listdir(rgb_dir):
        if (j > 200):
            img = np.load(rgb_dir + f)
            new_rgb_image = resize(img, (new_size[0], new_size[1]), anti_aliasing=True)
            new_rgb_file = des_dir + f[:15] + "_reshaped" + ".npy"

            for d in os.listdir(depth_dir):
                if f[:6] in d:
                    d_img = np.load(depth_dir + d)
                    new_depth_image = resize(d_img, (new_size[0], new_size[1]), anti_aliasing=True)
                    new_depth_file = des_dir + d[:15] + "_reshaped" + ".npy"
                    np.save(new_rgb_file, new_rgb_image)
                    np.save(new_depth_file, new_depth_image)
                    j += 1
                    break
        j +=1
        if (j > 205):
            break

#image scaling
def image_rescale(dirname,new_size):

    for f in os.listdir(dirname):
        img = np.load(dirname + f)
        # cv2.resize requires a tupe of w x h - crazy
        new_image = cv2.resize(img,(new_size[1],new_size[0]), interpolation=cv2.INTER_AREA)
        new_file = dirname+f[:15]+"_reshaped"+".npy"
        np.save(new_file,new_image)

# view scales image
def view_scaled_files(dirname):

    for f in os.listdir(dirname):
        if ("reshaped" in f):
            img = np.load(dirname + f)
            print("Shape = {}".format(img.shape))
            if ("depth" in f):
                plt.imshow(img,cmap= "gray")
            else:
                plt.imshow(img)
            plt.show()

def rotate_images(dirname):
    for f in os.listdir(dirname):
        img = np.load(dirname + f)
        plt.imshow("original", img)
        plt.show()

        angles = [15,25,40,65,80,100,120,140,175,200,225,265, 280,305,320]
        for i in angles:
            new_image = rotate(img,i)
        #new_image = cv2.resize(img,(new_size[1],new_size[0]), interpolation=cv2.INTER_AREA)
        #new_image = cv2.resize(img, (new_size[1], new_size[0]))
            plt.imshow("rotated at {}".format(i), new_image)
            plt.show()

# reading input files from directory and loading as N,B,G,R,Channels for RGB(X) and Depth(Y)
def load_data(dirname_rgb,dirname_d):

    myX = []
    myY = []

    for f in os.listdir(dirname_rgb):
        file_name = dirname_rgb + f
        if("reshaped" in file_name ):
            myX.append(np.load(file_name))

    for f in os.listdir(dirname_d):
        file_name = dirname_d + f
        if("reshaped" in file_name ):
            myY.append(np.load(file_name))

    X = np.asanyarray(myX)
    Y = np.asanyarray(myY)
    Ymask = Y.copy()

    Ymask[Ymask > 0] = 1
    Ymask = Ymask.astype('float32')
    X = X.astype('float32')
    Y = Y.astype('float32')

    return(X, Y, Ymask)

def load_10_data(dirname_rgb,dirname_d):
    myX = []
    myY = []

    j = 0

    for f in os.listdir(dirname_rgb):
        if (j<10):
            rgbfile_name = dirname_rgb + f

            for d in os.listdir(dirname_d):
                if f[:6] in d:
                    dfile_name = dirname_d + d
                    myX.append(np.load(rgbfile_name))
                    myY.append(np.load(dfile_name))
                    j+=1
                    break
        else:
            break
    return (np.asanyarray(myX), np.asanyarray(myY))


def load_test_data(dirname_rgb,dirname_d):

    myX = []
    myY = []

    for f in os.listdir(dirname_rgb):
        file_name = dirname_rgb + f
        if("reshaped" in file_name ):
            myX.append(np.load(file_name))

    for f in os.listdir(dirname_d):
        file_name = dirname_d + f
        if("reshaped" in file_name ):
            myY.append(np.load(file_name))

    return(np.asanyarray(myX), np.asanyarray(myY))


# Given an N images of 3 channels, it will scale and normalize the image
# to mean 0 and std 1 across the channels separately.
def feature_normalize_rgb(x):

    c1 = x[:,:,:,0]
    c2 = x[:,:,:,1]
    c3 = x[:,:,:,2]

    p = np.zeros(x.shape,np.float32)

    c1 = c1/255.0
    c1_mean = np.mean(c1,axis=0)
    c1_std = np.std(c1,axis=0)
    c1_myscaled = (c1-c1_mean)/c1_std
    p[:, :, :, 0] = c1_myscaled

    c2 = c2 / 255.0
    c2_mean = np.mean(c2, axis=0)
    c2_std = np.std(c2, axis=0)
    c2_myscaled = (c2 - c2_mean) / c2_std
    p[:, :, :, 1] = c2_myscaled

    c3 = c3 / 255.0
    c3_mean = np.mean(c3, axis=0)
    c3_std = np.std(c3, axis=0)
    c3_myscaled = (c3 - c3_mean) / c3_std
    p[:, :, :, 2] = c3_myscaled

    return p


# count the n pixels with non-zero depth for the N depth images
def count_pixels_with_depth(y):
    num = y.shape[0]
    N = []

    for i in range(num):
        count = 0
        for j in range(y.shape[1]):
            for k in range(y.shape[2]):
                if y[i, j, k] == 1:
                    count += 1
        N.append(count)
    return N


def depth_loss(y_true, y_pred):

    d = y_pred - y_true




def image_detail(dirname):

    for f in os.listdir(dirname):
        amax = 0
        amax_loc = [240,320]
        amin = 255
        img = np.load(dirname + f)
        print(img.shape)
        a = img[20,20]
        print("a = {}".format(a))
        if "depth" in f:
            k = 0
            my_tup = img.shape
            h = range(0,my_tup[0],1)
            w = range(0,my_tup[1],1)
            for i in h:
                print("{}".format(img[i,:]))
            for i in h:
                for j in w:
                    a = img[i,j]

                    # remove the large a value near the first two pixels
                    if a > 25000:
                        a=0

                    # span from 0 to 255
                    a = int(a/255)

                    # identify the spread maximum first
                    if a > amax:
                        amax = a
                        amax_loc[0]=i
                        amax_loc[1]=j

                    #For better visibility convert all the spots for which we have no depth to white color
                    if a == 0:
                        a = 255
                        k += 1

                    if a < amin:
                        amin = a

                    img[i,j] = a

            zero_percent = k*100/(len(w)*len(h))
            print("amin = {}, Zero Percentage = {}".format(amin,zero_percent))
            print("amax = {} at location {}".format(amax,amax_loc))

            for i in h:
                for j in w:
                    a = img[i,j]
                    if 5<a<100:
                        a = int(a*200/95)
                        img[i,j] = a

        else:
            pass

        #for i in h:
            #print("{}".format(img[i, :]))
            #p = 2

        plt.imshow(img, cmap='gray')
        plt.show()
        e= 10

dirname1 = "/home/p4bhattachan/PycharmProjects/syde770/images/rgb100/"
dirname2 = "/home/p4bhattachan/PycharmProjects/syde770/images/depth100/"

if __name__ == "__main__":

    #move(source,destination,"depth")
    #copy("/home/p4bhattachan/PycharmProjects/syde770/images/rgb/","/home/p4bhattachan/PycharmProjects/syde770/images/rgb100/")
    #copy("/home/p4bhattachan/PycharmProjects/syde770/images/depth/","/home/p4bhattachan/PycharmProjects/syde770/images/depth100/")

    #dirname = "/home/p4bhattachan/PycharmProjects/syde770/images/rgb100/"
    #dirname = "/home/p4bhattachan/PycharmProjects/syde770/images/depth100/"
    #image_detail(dirname)
    image_rescale("/home/p4bhattachan/PycharmProjects/syde770/images/test_depth/",(56,76))
    #view_scaled_files(dirname)
    #rotate_images(dirname)

    copy_and_rescale_images_for_testing("/home/p4bhattachan/PycharmProjects/syde770/images/rgb/","/home/p4bhattachan/PycharmProjects/syde770/images/depth/",(120,160))

    # load data
    X_train, Y_train = load_data(dirname1, dirname2)
    # Plot the first image
    plt.imshow(X_train[0])
    plt.show()

    X_train = X_train.astype('float32')

    p = feature_normalize_rgb(X_train)

    n = p[0].shape


    exit(1)



    #X_train = X_train.astype('float32')
    #Y_train = Y_train.astype('float32')

    # define data preparation
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,data_format="channels_last",rescale=1.0/255.0)
    # fit parameters from data
    datagen.fit(X_train)
    # configure batch size and retrieve one batch of images
    for X_batch, y_batch in datagen.flow(X_train, Y_train, batch_size=10):
        for i in range(0, 10):
            f = X_batch[i].shape
            #plt.imshow(X_batch[i])
            #show the plot
            #plt.show()
        break
    print(len(X_batch[0][0, :, 0]))
    print(X_batch[0][0, :, 0])

