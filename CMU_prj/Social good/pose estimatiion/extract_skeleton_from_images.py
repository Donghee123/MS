from ntpath import join
import cv2
import mediapipe as mp
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:

def sort_imageNmae(listOfImage_Path : list):
  listOfsortedImagePath = []
  for nIndex,str_Image_Path in enumerate(listOfImage_Path):
    newstring = ''.join([i for i in str_Image_Path if not i.isdigit()])
    index = newstring.find('.jpg') 
    newstring = newstring[:index] + str(nIndex) + newstring[index:]
    listOfsortedImagePath.append(newstring)
  return listOfsortedImagePath

def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise

BG_COLOR = (192, 192, 192) # gray

def GetPos(image : np.ndarray):
  with mp_pose.Pose(
      static_image_mode=True,
      model_complexity=2,
      enable_segmentation=True,
      min_detection_confidence=0.5) as pose:
    return pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def GetCVImages(label, strfilelist : list ):
  
  listOfImages = []

  for str_Image_Path in strfilelist:
      str_Image_file = os.path.join(str_folder_path, label, str_Image_Path)
      listOfImages.append(cv2.imread(str_Image_file))

  return listOfImages

if __name__ == '__main__':
  bUseTest = True

  str_folder_path = 'sample_image_folder'
  folderLabels = ['right','left','good','turtleneck']

  if bUseTest == True:
    str_folder_path = 'for_presentation'
    folderLabels = ['testvideo']

  for folderLabel in folderLabels:
    with mp_pose.Pose(
          static_image_mode=True,
          model_complexity=2,
          enable_segmentation=True,
          min_detection_confidence=0.5) as pose:
        
        listOfImagesPath = os.listdir(os.path.join(str_folder_path, folderLabel))
        listOfImagesPath = sort_imageNmae(listOfImagesPath)
        listOfImages = GetCVImages(folderLabel, listOfImagesPath)

        for nIndex, cvimage in enumerate(listOfImages):
          
          results = GetPos(cvimage)
          image_height, image_width, _ = cvimage.shape
          if not results.pose_landmarks:
            continue
          print(
              f'Nose coordinates: ('
              f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
              f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
          )
        
          annotated_image = cvimage.copy()
          # Draw segmentation on the image.
          # To improve segmentation around boundaries, consider applying a joint
          # bilateral filter to "results.segmentation_mask" with "image".
          condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
          bg_image = np.zeros(cvimage.shape, dtype=np.uint8)
          bg_image[:] = BG_COLOR
          annotated_image = np.where(condition, annotated_image, bg_image)
          # Draw pose landmarks on the image.
          mp_drawing.draw_landmarks(
              annotated_image,
              results.pose_landmarks,
              mp_pose.POSE_CONNECTIONS,
              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
          
          str_save_folder = os.path.join(f'{str_folder_path}',f'skeleton',f'{folderLabel}')
          str_npy_save_folder = os.path.join(f'{str_folder_path}',f'skeleton_npy',f'{folderLabel}')

          makedirs(str_save_folder)
          makedirs(str_npy_save_folder)
          
          cv2.imwrite(os.path.join(str_save_folder,f'{folderLabel}_{nIndex}.jpg'), annotated_image)

          listOfOneSheleton = []

          for landmark in results.pose_landmarks.landmark:
              listOfOneSheleton.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

          npOfLandMarks = np.array(listOfOneSheleton)

          assert len(npOfLandMarks) == 33, 'All landmarks size must be 33'
          np.save(os.path.join(str_npy_save_folder,f'{folderLabel}_{nIndex}'), npOfLandMarks)
          
        # Plot pose world landmarks.
          #mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS, savepath = os.path.join(str_3D_save_folder, f'{folderLabel}_{nIndex}'))
