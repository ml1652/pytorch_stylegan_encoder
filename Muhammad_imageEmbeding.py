import pandas as pd
import numpy as np
import json

import requests
url = 'https://cvssp.org/projects/facer2vm/fverify/px.php?dst=/verify_api'

data = pd.read_table("/scratch/staff/ml1652/StyleGAN_Reconstuction_server/celeba/Anno/list_attr_celeba.txt", skiprows=1, header=0, sep=r"\s+")
names = data.index

image_directory = "/scratch/staff/ml1652/StyleGAN_Reconstuction_server/celeba/img_celeba"

filenames = [f"{image_directory}/{x}" for x in names]
descriptors = []

filenames = filenames[0:10]
d = {}
count = 0
for image_file in filenames:
    img = open(image_file, 'rb')
    file_obj = {}
    file_obj['img1'] = img
    data_obj = {}
    data_obj['img1_data'] = '0,0,0,0'
    data_obj['qfeat_only'] = 'true'
    data_obj['apikey'] = 'KIoonmaslanlebw3XyoZ5Cqke09E'
    x = requests.post(url, files=file_obj, data=data_obj)


    try:
        resp = json.loads(x.text)
    except  Exception:
        print(f'{image_file} failed: failed to parse resp - {x.text}')
        count += 1
        continue

    if 'status' in resp and resp['status'] == 'Error':
        print(f'{image_file} failed: {resp["message"]}')
        count += 1
        continue
    split = image_file.split('/')[-1]
    img_name = split.split('\\')[-1]
    features = np.array(resp['query_feat'])
    d[img_name] = features
    #descriptors.append(d)


    print(f'{image_file} ok')

save_path = '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/face_enbeding_skipped_img_='+str(count)+'.npy'

np.save(save_path, d)
