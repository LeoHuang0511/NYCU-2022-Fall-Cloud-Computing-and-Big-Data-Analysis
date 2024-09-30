import glob
import os
from moviepy.editor import *
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_data_path", type=str, default="./data", help="original data path")
    parser.add_argument("--path_to_save", type=str, default="./data_resized", help="path for saving resized data")
    info = parser.parse_args()

    train_dir = info.ori_data_path+'/train'
    test_dir = info.ori_data_path+'/test'
    category_dir_list = os.listdir(train_dir) 
    print('categories amount:',len(category_dir_list))
    train_list = []
    train_label_list = []
    test_list = []
    for cat in category_dir_list:
        train_list.append(glob.glob(os.path.join(train_dir+'/'+cat,'*.mp4'))) 
        for i in range(len(glob.glob(os.path.join(train_dir+'/'+cat,'*.mp4')))) :
            train_label_list.append(cat)
    test_list = glob.glob(os.path.join(test_dir,'*.mp4'))
    #print(train_list)
    train_list = [i for item in train_list for i in item]
    print('train data amount:',len(train_list))
    print('label list amount:', len(train_label_list))
    #print(test_list)
    print('test data amount:',len(test_list))

    resize_path = info.path_to_save
    os.makedirs(resize_path+"/train", exist_ok=True)
    os.makedirs(resize_path+"/test", exist_ok=True)
    for i in range(39):
        os.makedirs(resize_path+f"/train/{i}", exist_ok=True)
    for i in range(len(train_list)):
        video = VideoFileClip(train_list[i])
        filename = os.path.split(train_list[i])[1]
        #print(filename)
        output = video.resize((256,256))
        output.write_videofile(resize_path+f"/train/"+train_label_list[i]+"/"+filename,fps=8)

    for i in range(len(test_list)):
        video = VideoFileClip(test_list[i])
        filename = os.path.split(test_list[i])[1]
        #print(filename)
        output = video.resize((256,256))
        output.write_videofile(resize_path+f"/test/"+filename,fps=8)

    category_dir_list_resize = os.listdir(resize_path+"/train") 
    for cat in category_dir_list_resize:
        video_each_cat = os.listdir(resize_path+"/train/"+cat) 
        print(cat+f":{len(video_each_cat)}")
    test_list_resize = os.listdir(resize_path+"/test") 
    print("test videos number: ", len(test_list_resize))


if __name__ == '__main__':
    main()