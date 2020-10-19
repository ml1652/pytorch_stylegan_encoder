import cv2
import os
import numpy as np
from natsort import natsorted

images = []



# def count_datasets_num (directory):
#     num = 0
#     for filename in os.listdir(directory):
#         if os.path.splitext(filename)[1] == '.jpg':
#             num +=1
#
#     return num
#
# orginal_image_num = count_datasets_num(directory)

original_directory = "C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/AFLW_vggface2"


flip_datasets = 'C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/AFLW_vggface2_tightcrop_flip'

#enlarge the datasets by horizehntal flip images
def flip_image():
    flipped_paths = []
    original_paths = []
    #create new datasets's folder
    if not os.path.exists(flip_datasets):
        os.makedirs(flip_datasets)

    file_list = list(os.listdir(original_directory))
    file_list = natsorted(file_list)

    for filename in file_list:
        # print(f'processing {filename}')
        if os.path.splitext(filename)[1] != '.jpg':
            continue
        if filename.startswith('flip_'):
            continue

        original_path = os.path.join(original_directory,filename)
        img = cv2.imread(original_path)
        output_path = os.path.join(flip_datasets,'flip_' + filename)
        image_flip = cv2.flip(img, 1)
        cv2.imwrite(output_path, image_flip)
        #original_paths.np.loadappend(original_path)
        flipped_paths.append(output_path)
        #save_path = 'A:/test2/path.npy'
        #np.save(save_path ,flipped_paths)
    return flipped_paths


#pose
def pose_extend(paths):
    poses = np.load("D:/AFLW/numpylist2/pose_data_with_name.npz", allow_pickle=True)
    filenames = poses['path']
    old_path = []
    for path in filenames:
        old_path.append(path)

    for flip_path in paths:
        old_path.append(flip_path)

    path_new = old_path
    negate_pose = []
    pose_sets = np.stack(poses['pose'])
    for pose in pose_sets:
        negate_pose.append(pose)
    negate_pose = np.stack(negate_pose)
    negate_pose[:, 2] = -(negate_pose[:, 2])

    output = np.concatenate((np.stack(pose_sets), negate_pose))

    save_path = ("D:/AFLW/numpylist2/pose_data_with_name_flip.npz")
    np.savez(save_path, path=path_new, pose=output)


pose_extend(flip_image())











