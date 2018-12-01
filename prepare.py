
import os
import numpy as np
import dummy
import shutil


from skimage.transform import rescale, resize, downscale_local_mean





def build_RS_rgbfiles(source_dir, destination_dir, match_str1,match_str2):
    i = 0
    j = 0

    for f in os.listdir(source_dir):
        if match_str1 in f:
            filename = source_dir + f
            img = np.load(filename)
            new_image = resize(img, (120, 160), anti_aliasing=True)
            new_file = destination_dir +"rgb/"+f[:15] + "_reshaped" + ".npy"
            np.save(new_file, new_image)
            i += 1
        elif match_str2 in f:
            filename = source_dir + f
            img = np.load(filename)
            new_image = resize(img,(56, 76), anti_aliasing=True)
            new_file = destination_dir + "depth/" + f[:15] + "_reshaped" + ".npy"
            np.save(new_file, new_image)
            j +=1

    print("total number rgb files = {}  depth files = {}".format(i,j))


def load_more_data(dirname_rgb, dirname_d, k):
    myX = []
    myY = []
    i = 0
    j = 0

    for f in os.listdir(dirname_rgb):
        rgbfile_name = dirname_rgb + f
        match = 0
        for h in os.listdir(dirname_d):
            depthfile_name = dirname_d + h
            if (f[:6] in depthfile_name):
                myY.append(np.load(depthfile_name))
                myX.append(np.load(rgbfile_name))
                i += 1
                match = 1
                break
        if match == 0:
            j += 1
        if i > k:
            break

    print("Total match {} - total unmatched {}".format(i-1, j))

    X = np.asanyarray(myX)
    Y = np.asanyarray(myY)
    Ymask = Y.copy()

    Ymask[Ymask > 0] = 1
    Ymask = Ymask.astype('float32')
    X = X.astype('float32')
    Y = Y.astype('float32')

    return (X, Y, Ymask)



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


def move_test_data(dirname_rgb, dirname_d, dest_dir, k):

    i=0
    for f in os.listdir(dirname_rgb):
        rgbfile_name = dirname_rgb + f
        for h in os.listdir(dirname_d):
            depthfile_name = dirname_d + h
            if (f[:6] in depthfile_name):
                shutil.move(rgbfile_name, dest_dir+"test_rgb")
                shutil.move(depthfile_name, dest_dir + "test_depth")
                i += 1
                break
        if i >= k:
            break

    print("File pairs moved = {}".format(i))



def load_test_data(dirname_rgb,dirname_d):

    myX = []
    myY = []

    i = 0
    for f in os.listdir(dirname_rgb):
        rgbfile_name = dirname_rgb + f
        for h in os.listdir(dirname_d):
            depthfile_name = dirname_d + h
            if (f[:6] in depthfile_name):
                myX.append(np.load(rgbfile_name))
                myY.append(np.load(depthfile_name))
                i += 1
                break


    print("Test File pairs loaded {}".format(i-1))

    return(np.asanyarray(myX), np.asanyarray(myY))




if __name__ == "__main__":

    #build_RS_rgbfiles("/home/p4bhattachan/gripper/Data/Images/","/home/p4bhattachan/PycharmProjects/syde770/images/","RS_color.npy","RS_depth.npy")

    #load_more_data("/home/p4bhattachan/PycharmProjects/syde770/images/rgb/","/home/p4bhattachan/PycharmProjects/syde770/images/depth/",250)

    move_test_data("/home/p4bhattachan/PycharmProjects/syde770/images/rgb/","/home/p4bhattachan/PycharmProjects/syde770/images/depth/","/home/p4bhattachan/PycharmProjects/syde770/images/",500)
