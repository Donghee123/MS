import cv2
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm


def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise

if __name__ == '__main__':
    str_videos_folder = 'videos'
    str_fileExt = ".mp4"

    str_videos_names = [_ for _ in os.listdir(str_videos_folder) if _.endswith(str_fileExt)]

    #str_videos_names = ['good.mp4','left.mp4','right.mp4','turtleneck.mp4']
    str_videos_names = ['testvideo.mp4']
    for str_videos_name in tqdm(str_videos_names):

        str_origin_file_name = Path(str_videos_name).stem
        str_save_directory = os.path.join (f'{str_videos_folder}',f'{str_origin_file_name}')

        makedirs(str_save_directory)

        vidcap = cv2.VideoCapture(os.path.join(f'{str_videos_folder}',f'{str_videos_name}'))

        count = 0 

        while(vidcap.isOpened()): 

            ret, image = vidcap.read() 

            if type(image) != np.ndarray:
                break
            
            if(int(vidcap.get(1)) % 1 == 0):
                try: 
                    cv2.imwrite(os.path.join(f'{str_save_directory}',f'{str_origin_file_name}_{count}.jpg'), image)
                except:
                    break
                count += 1
            

        vidcap.release()
