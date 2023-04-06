import os
import io
import json
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import requests
import cv2

"""
| Attribute Type List
| https://docs.microsoft.com/en-us/python/api/azure-cognitiveservices-vision-face/azure.cognitiveservices.vision.face.models.faceattributetype?view=azure-python
"""

credential = json.load(open('AzureCloudKeys.json'))
API_KEY = credential['API_KEY']
ENDPOINT = credential['ENDPOINT']

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))
path=r"IMG_PATH"
image_file=open(path,'rb')
response_detection = face_client.face.detect_with_stream(
    image=image_file,
    detection_model='detection_01',
    recognition_model='recognition_03',
    return_face_attributes=['age', 'emotion'],
)

frame=cv2.imread(path)


for face in response_detection:
    age = face.face_attributes.age
    emotion = face.face_attributes.emotion
    neutral = '{0:.0f}%'.format(emotion.neutral * 100)
    happiness = '{0:.0f}%'.format(emotion.happiness * 100)
    anger = '{0:.0f}%'.format(emotion.anger * 100)
    sandness = '{0:.0f}%'.format(emotion.sadness * 100)

    rect = face.face_rectangle
    left = rect.left
    top = rect.top
    right = rect.width + left
    bottom = rect.height + top
    cv2.rectangle(frame,(left,top),(right,bottom),color=(255,0,100),thickness=2,)
    cv2.putText(frame,'Age: ' + str(int(age)),(right + 4, top),cv2.FONT_HERSHEY_DUPLEX,1.0,(125, 246, 55), 2)
    cv2.putText(frame, 'Neutral: ' + neutral,(right + 4, top+35),cv2.FONT_HERSHEY_DUPLEX,1.0,(125, 246, 55), 2)
    cv2.putText(frame, 'Happy: ' + happiness,(right + 4, top+70),cv2.FONT_HERSHEY_DUPLEX,1.0,(125, 246, 55), 2)
    cv2.putText(frame, 'Sad: ' + sandness,(right + 4, top+105),cv2.FONT_HERSHEY_DUPLEX,1.0,(125, 246, 55), 2)
    cv2.putText(frame, 'Angry: ' + anger,(right + 4, top+140),cv2.FONT_HERSHEY_DUPLEX,1.0,(125, 246, 55), 2)

cv2.imshow('face',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()