from concurrent.futures import ThreadPoolExecutor
import imghdr
import shutil

import PIL
from bing_image_downloader import downloader
from bing_images import bing
import os
import glob
import time
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from PIL import Image

def scrape_images(query_list):
    for query in query_list:
        # Make output directory
        parent_dir = "D:/SE2023-9.1/SDmodel/data/train/"
        #output_path = os.path.join(parent_dir, query)
        #os.makedirs(output_path, exist_ok=True)
        
        # Download images using bing_image_downloader
        downloader.download(query , limit=1500, output_dir=parent_dir, 
                            adult_filter_off=True, force_replace=False, timeout=1800, verbose=True)

def scrape_multithreads(query_list):
    output_dir = "D:/SE2023-9.1/SDmodel/data/train/Human_face_test"
   
    list = ['old personface']
    for query in list:
        #output_dir = os.path.join(parent_dir, f"{query}")
        query_dir = os.path.join(output_dir, query.replace(" ", "_"))
        os.makedirs(query_dir, exist_ok=True)
        
        bing.download_images(query,
                            200,
                            output_dir=query_dir,
                            pool_size=10,
                            file_type = 'jpg',
                            force_replace=False,
                            extra_query_params='&first=1'
                            )
        time.sleep(1)
        
    # copy all image into 1 folder human_face
    #newdir = "D:/SE2023-9.1/SDmodel/data/train/Human_face"
    #os.makedirs(newdir, exist_ok=True)
    #img_list = glob.glob('D:/SE2023-9.1/SDmodel/data/train/Human_face_test/*/*')
    #for img in tqdm(img_list):
    #    shutil.copy2(img, newdir)

def rename_img():
    image_path = "D:/SE2023-9.1/SDmodel/data/train"    #### provide path where image is stored

    dir_list = glob.glob(image_path+'/Human_face_test/*')
    #print(filenames)

    for i, dir in enumerate(dir_list):
        file_list = glob.glob(image_path+'/Human_face_test/*/*')
    #ogfile = glob.glob(image_path+'*/*')
    for j, file in tqdm(enumerate(file_list)):
        new_path = os.path.join(os.path.dirname(file), f'Image_13_{i}_{j}.jpg')   
        # Check if the target file already exists
        if not os.path.exists(file):
            continue
        if os.path.exists(new_path):
            continue  # Skip the renaming if the file already exists
        # Rename the file
        os.rename(file, new_path)

    nimgs = glob.glob(image_path+'/Human_face_test/*/*')
    for img in tqdm(nimgs):
        try:
            img_format = imghdr.what(img)
            format_list = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']
            
            if img_format not in format_list:
                os.remove(img)
            
            image = Image.open(img).convert('RGBA')
            image.close()
        except (PIL.UnidentifiedImageError, OSError):
            if os.path.exists(img):
                os.remove(img)
            continue
'''
def download_image(url, output_path):
    response = requests.get(url)
    with open(output_path, "wb") as file:
        file.write(response.content)


def rename_file(src_path, dst_path):
    max_retries = 5
    retry_delay = 0.1

    for _ in range(max_retries):
        try:
            os.rename(src_path, dst_path)
            break
        except PermissionError:
            time.sleep(retry_delay)
    else:
        print(f"Failed to rename file: {src_path}")

def scrape_multithreads(query_list2):
    parent_dir = "D:/SE2023-9.1/data/train/"
   
    #os.makedirs(parent_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=10) as executor:
        for query in query_list2:
            output_dir = os.path.join(parent_dir, f"{query}")
            os.makedirs(output_dir, exist_ok=True)
            
            bing.download_images(query, 100, output_dir=output_dir, pool_size=10, file_type='jpg',
                                 force_replace=True, extra_query_params='&first=1')
            
            for file_name in os.listdir(output_dir):
                if file_name.startswith("#tmp#"):
                    src_path = os.path.join(output_dir, file_name)
                    dst_path = os.path.join(output_dir, file_name[6:])
                    executor.submit(rename_file, src_path, dst_path)
'''

def modifiedImg():
    img_path = glob.glob('D:/SE2023-9.1/SDmodel/data/train/Human_face_test/*/*')
    
    # load image
    for path in img_path:
        img = cv2.imread(path)
        #img_name = os.path.basename(path).split('/')[-1]

        # convert to graky
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # threshold input image as mask
        mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]

        # negate mask
        mask = 255 - mask

        # apply morphology to remove isolated extraneous noise
        # use borderconstant of black since foreground touches the edges
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # anti-alias the mask -- blur then stretch
        # blur alpha channel
        mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

        # linear stretch so that 127.5 goes to 0, but 255 stays 255
        mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

        # put mask into alpha channel
        result = img.copy()
        result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = mask

        # save resulting masked image
        cv2.imwrite(path, result)

if __name__ == "__main__":
    query_list = ["Vietnam places", "Vietnam people", "Vietnam culture", 'Vietnam traditions', 'Vietnam food']
    query_list1 = ['Vietnam history', 'Vietnam architure', 'Vietnam art', 'Vietnam travel', 'Vietnam conical hat people', 'Vietnam ao dai', 'Vietnam clothes']
    query_list2 = ['Asian_face']
    #scrape_images(query_list)
    #scrape_images(query_list1)
    #scrape_multithreads(query_list)
    #scrape_multithreads(query_list2)
    rename_img()
    modifiedImg()