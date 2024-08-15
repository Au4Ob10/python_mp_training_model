import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import json
import os
import re



model_path = './models/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
    
)

with HandLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file('../ChicagoFSWild/deafvideo_3/deaf_power_rob_3089/0002.jpg')
        hand_landmarker_result = landmarker.detect(mp_image)
        
        print(hand_landmarker_result)
        
        # stringified_result = str(hand_landmarker_result)
        
        # result_pattern = r'NormalizedLandmark\((.*?)\)'
        
        # matches = re.findall(stringified_result,result_pattern)
        
        # landmark_list = []
        
        # for match in matches:
            
        #     landmark_dict = {}
            
        #     for pair in match.split(', '):
        #         key, value = pair.split('=')
        #         landmark_dict[key] = float(value)
        #     landmark_list.append(landmark_dict)
            
        # json_data = json.dumps(landmark_list, indent=4)
        
        # print(json_data)

        
        
        
       
        
        
        # with open('sample.json', "w", encoding='utf8') as json_file:
        #     for lm_num, lm_val in enumerate(results_dict['hand_world_landmarks'][0][:]):
                
        #         replacements = {"(":"{", ")" : "}", "=" : ":", "x": "'x", "y": "'y", "z": "'z", ":": "':"}
                
        #         landmark_coor = (str(lm_num  + 1) + "_" + str(lm_val))
                
        
        #         for prev, new in replacements.items():
        #             landmark_coor = landmark_coor.replace(prev, new)
                
            
                
            
        #         json.dump(landmark_coor, json_file)
            
       
        
# with open("sample.json", "w") as outfile:  
#     json.dumps(hand_landmarker_result)
        
    
    
   







