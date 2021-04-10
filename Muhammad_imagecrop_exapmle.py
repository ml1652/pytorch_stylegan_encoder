import pandas as pd
import numpy as np
import json
import requests
import base64
import cv2

url = 'https://cvssp.org/projects/facer2vm/fverify/px.php?dst=/verify_api'
data = pd.read_table("C:/Users/Mingrui/Desktop/celeba/Anno/list_attr_celeba.txt", skiprows=1, header=0, sep=r"\s+")
names = data.index
image_directory = "C:/Users/Mingrui/Desktop/celeba/img_celeba"
filenames = [f"{image_directory}/{x}" for x in names]
descriptors = []
filenames = filenames[0:11]
d = {}
count = 0

for image_file in filenames:
    img = open(image_file, 'rb')
    file_obj = {}
    file_obj['img1'] = img
    data_obj = {}
    data_obj['apikey'] = 'F2VMl0cintuS3'
    data_obj['qfeat_only'] = 'true'
    data_obj['img1_cropped'] = 'false'
    data_obj['aligned_face'] = 'true'
    x = requests.post(url, files=file_obj, data=data_obj)

    try:
        resp = json.loads(x.text)
        aligned_face = resp['aligned_face']

        #jpg_as_text = resp['query_feat'].text
        #jpg_as_text = x.text
        jpg_original = base64.b64decode(aligned_face)
        image_save = image_file.split('/')[-1]
        im_arr = np.frombuffer(jpg_original, dtype=np.uint8)  # im_arr is one-dim Numpy array
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.flip(img, 1)
        cv2.imwrite(image_save, img)

        # with open(image_save, 'wb') as f_output:
        #     #f_output.write(jpg_original)


    except  Exception:
        print(f'{image_file} failed: failed to parse resp - {x.text}')
        count += 1
        continue
