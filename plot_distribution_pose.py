
import numpy as np
import sqlite3
import matplotlib.pyplot as plt

from PIL import ImageFile

def plot_original_datasets_pose_distribution():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    database_path = 'D:/AFLW/aflw/data/aflw.sqlite'
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    pose = []

    select_string = "faceimages.filepath, faces.face_id, facepose.roll, facepose.pitch, facepose.yaw, facerect.x, facerect.y, facerect.w, facerect.h, facemetadata.sex"
    from_string = "faceimages, faces, facepose, facerect, facemetadata"
    where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id and faces.face_id = facemetadata.face_id"

    query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string
    for row in c.execute(query_string):
        roll = row[2]
        pitch = row[3]
        yaw = row[4]
        pose.append(np.asarray([roll, pitch, yaw]))
    pose=np.array(pose)

    # plt.xlim(-0.5, 0.5)
    # plt.hist(pose[:,0], bins='auto')
    # plt.title("Origianl_datasets_roll_distribution")
    # plt.savefig("Origianl_datasets_roll_distribution")
    # plt.show()
    # plt.xlim(-0.5, 0.5)
    # plt.hist(pose[:,1], bins='auto')
    # plt.title("Origianl_datasets_pitch_distribution")
    # plt.savefig("Origianl_datasets_pitch_distribution")
    # plt.show()
    # plt.xlim(-0.5, 0.5)
    # plt.hist(pose[:,2], bins='auto')
    # plt.title("Origianl_datasets_yaw_distribution")
    # plt.savefig("Origianl_datasets_yaw_distribution")
    # plt.show()

    plt.figure(figsize = (20,10))
    plt.suptitle("datasets_pose_distribution")
    plt.subplot(2,3,1)
    plt.xlim(-1.5, 1.5)
    plt.hist(pose[:,0], bins='auto',color = 'g')
    plt.title("Origianl_datasets_roll_distribution")

    plt.subplot(2,3,2)
    plt.xlim(-1.5, 1.5)
    plt.hist(pose[:,1], bins='auto',color = 'g')
    plt.title("Origianl_datasets_pitch_distribution")
    plt.subplot(2,3,3)
    plt.xlim(-2.5, 2.5)
    plt.hist(pose[:,2], bins='auto',color = 'g')
    plt.title("Origianl_datasets_yaw_distribution")
    #plt.savefig("Origianl_datasetsdistribution")

    #load the mtcnn processed pose data
    porcessed_pose = np.load("D:/AFLW/numpylist2/pose_data.npy")
    plt.subplot(2, 3, 4)
    plt.xlim(-1.5, 1.5)
    plt.hist(porcessed_pose[:, 0], bins='auto')
    plt.title("Processed_roll_distribution")

    plt.subplot(2, 3, 5)
    plt.xlim(-1.5, 1.5)
    plt.hist(porcessed_pose[:, 1], bins='auto')
    plt.title("Processed_pitch_distribution")
    plt.subplot(2, 3, 6)
    plt.xlim(-2.5, 2.5)
    plt.hist(porcessed_pose[:, 2], bins='auto')
    plt.title("Processed_yaw_distribution")
    plt.savefig("datasets_pose_distribution")

plot_original_datasets_pose_distribution()