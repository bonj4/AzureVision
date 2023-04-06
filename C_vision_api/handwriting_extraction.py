import os
import io
import json
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes,VisualFeatureTypes
import requests
import cv2
import time

credential = json.load(open('C_vision_api/AzureCloudKeys.json'))
API_KEY = credential['API_KEY']
ENDPOINT = credential['ENDPOINT']

cv_client = ComputerVisionClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))

path=r"C_vision_api\sample.png"
image_file=open(path,'rb')
response = cv_client.read_in_stream(image=image_file,raw=True)
read_operation_location = response.headers["Operation-Location"]
# Grab the ID from the URL
operation_id = read_operation_location.split("/")[-1]

# Call the "GET" API and wait for it to retrieve the results 
while True:
    read_result = cv_client.get_read_result(operation_id)
    if read_result.status not in ['notStarted', 'running']:
        break
    time.sleep(1)
frame=cv2.imread(path)

result=cv_client.get_read_result(operation_id)
# print(result.analyze_result)
if result.status==OperationStatusCodes.succeeded:
    read_result=result.analyze_result.read_results
    # print(read_result)
    for analyzed_result in read_result:
        for line in analyzed_result.lines:
            print(line.text)
            x1,y1,x2,y2,x3,y3,x4,y4=line.bounding_box
            # frame=cv2.rectangle(frame,(x1,y1),(x2,y2),color=(255,255,255),thickness=1)
            print(x1,y1,x2,y2,x3,y3,x4,y4)

            image = cv2.rectangle(frame, (int(x1),int(y1)), (int(x3),int(y3)), (0,0,255), 2)
image=cv2.resize(image,(500,500))
cv2.imshow('winname',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
