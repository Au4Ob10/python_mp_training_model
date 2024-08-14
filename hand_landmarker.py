# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("image.jpg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))







# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import numpy as np
# import cv2
# import os


# model_path = './models/hand_landmarker.task'

# BaseOptions = mp.tasks.BaseOptions
# HandLandmarker = mp.tasks.vision.HandLandmarker
# HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode

# options = HandLandmarkerOptions(
#     base_options = BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.IMAGE
    
# )

# with HandLandmarker.create_from_options(options) as landmarker:
#     mp_image = cv2.imread('../ChicagoFSWild/youtube_2/alex_abenchuchan_3297.jpg')
#     image_rgb = cv2.cvtColor(mp_image,  cv2.COLOR_BGR2RGB)
#     hand_landmarker_result = landmarker.process(image_rgb)
    
#     if hand_landmarker_result.multi_hand_landmarks:
#         for hand_landmarks in hand_landmarker_result:
#             print("Hand landmarks detected:")
#             for idx, landmark in enumerate(hand_landmarks.landmark):
#                 x = landmark.x
#                 y = landmark.y
#                 z = landmark.z
#                 print(f"Landmark {idx}: x={x}, y={y}, z={z}")
#             mp.solutions.drawing_utils.draw_landmarks(mp_image , hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    
#     cv2.imshow('Hand Landmarks', mp_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()





# mp_solutions = mp.solutions.drawing_utils

# base_options = python.BaseOptions(model_asset_path='./models/hand_landmarker.task')

# options = vision.HandLandmarkerOptions(base_options=base_options, num_hands = 1)

# detector = vision.HandLandmarker.create_from_options(options)

# image  = mp.Image.create_from_file('../ChicagoFSWild/youtube_2/alex_abenchuchan_3297/0002.jpg')

# detection_result = detector.detect(image)

# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

# cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))





