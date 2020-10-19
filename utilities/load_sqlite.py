# from https://github.com/ashishtrivedi16/HyperFace/blob/master/src/preprocess.py

import os
import cv2
import numpy as np
import sqlite3
import dlib
from sklearn.model_selection import train_test_split
from mtcnn import FaceDetector
from PIL import Image, ImageDraw, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt

# Path variables
image_path = 'D:/AFLW/aflw/data/flickr/'
database_path = 'D:/AFLW/aflw/data/aflw.sqlite'
image_save_path = 'C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/AFLW_vggface2/'

detector = FaceDetector()


# remove bboxes that is too small
def detect_bboxes(bboxes):
    new_bboxes = []
    # the minimal size of bbooxes we want to keep
    min_size = 100
    for b in bboxes:
        box_w = b[2] - b[0]
        box_h = b[3] - b[1]

        if box_w > min_size or box_h > min_size:
            new_bboxes.append(b)
    return new_bboxes


def largest_bbox(bboxes):
    if len(bboxes) <= 1:
        return bboxes[0]

    highest_area = 0
    box = None

    for b in bboxes:
        box_w = b[2] - b[0]
        box_h = b[3] - b[1]
        area = box_h * box_w

        if area > highest_area:
            box = b
            highest_area = area
    return box


def get_face_count(image_path):
    image = Image.open(image_path)

    try:
        bboxes, _ = detector.detect(image)
        bboxes = detect_bboxes(bboxes)
        return len(bboxes)
    except Exception:
        return 0


def expand_image(image, padding):
    padding = tuple(map(lambda v: max(0, v), padding))
    return ImageOps.expand(image, padding)


def get_raw_data(image_path, database_path):
    images = []
    landmarks = []
    visibility = []
    pose = []
    gender = []
    paths = []
    paths_cropped = []

    # Image counter
    counter = 1
    grey_count = 0
    # Open the sqlite database
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    image_count = 0

    # Creating the query string for retriving: roll, pitch, yaw and faces position
    # Change it according to what you want to retrieve
    select_string = "faceimages.filepath, faces.face_id, facepose.roll, facepose.pitch, facepose.yaw, facerect.x, facerect.y, facerect.w, facerect.h, facemetadata.sex"
    from_string = "faceimages, faces, facepose, facerect, facemetadata"
    where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id and faces.face_id = facemetadata.face_id"
    # where_string = f'({where_string}) and faceimages.image_id in (16076, 16064, 16066, 39460)'
    #where_string = f'({where_string}) and faceimages.image_id > 19980 and faceimages.image_id < 20000'
    # where_string = f'({where_string}) and faceimages.image_id in (39460)'
    # where_string = f'({where_string}) and faceimages.image_id in (16285)'

    query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string

    # It iterates through the rows returned from the query
    for row in c.execute(query_string):

        # Using our specific query_string, the "row" variable will contain:
        # row[0] = image path
        # row[1] = face id
        # row[2] = roll
        # row[3] = pitch
        # row[4] = yaw
        # row[5] = face coord x
        # row[6] = face coord y
        # row[7] = face width
        # row[8] = face heigh
        # row[9] = sex

        # Creating the full path names for input and output

        input_path = image_path + str(row[0])
        # output_path = storing_path + str(row[0])

        # If the file exist then open it
        if not os.path.isfile(input_path):
            continue

        # #opencv version
        # image = cv2.imread(input_path, 1)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # load the colour version
        # PIL version
        image = Image.open(input_path)
        # image = Image.open('D:/AFLW/aflw/data/flickr/3/image02045.jpg')
        # image = Image.open('D:/AFLW/aflw/data/flickr/3/image00091.jpg')

        print(input_path)
        # image = Image.open("C:/Users/Mingrui/Desktop/vgg2-testimage/n000106/0002_01.jpg")

        if len(image.split()) < 3:
            print(f'image: {row[0]}，(skip: grey image)', end='')
            grey_count = grey_count + 1
            continue
        face_count = get_face_count(input_path)
        print(f'image: {row[0]}, face count: {face_count}', end='')
        if face_count > 1 or face_count == 0:
            print('(skip： mlutiple faces)')
            continue
        print(' (OK)')

        paths.append(input_path)

        # Image dimensions
        image_width, image_height = image.size

        # Roll, pitch and yaw
        roll = row[2]
        pitch = row[3]
        yaw = row[4]
        # Face rectangle coords
        face_x = row[5]
        face_y = row[6]
        face_w = row[7]
        face_h = row[8]
        # Gender
        sex = (1 if row[9] == 'm' else 0)
        gender.append(sex)
        pose.append(np.asarray([roll, pitch, yaw]))

        # bounding_boxes = detect_bboxes(bounding_boxes)

        # Run MTCNN on the image
        detector = FaceDetector()
        bounding_boxes, _ = detector.detect(image)

        bounding_boxes = [largest_bbox(bounding_boxes)]

        for b in bounding_boxes:
            b = list(map(lambda v: max(v, 0), b))
            face_x = int(b[0])
            face_y = int(b[1])
            face_w = int(b[2] - b[0])
            face_h = int(b[3] - b[1])

            # Resize the MTCNN facebox by 1.3:
            resize_ratio = 2
            face_w_resized = round(face_w * resize_ratio)
            face_h_resized = round(face_h * resize_ratio)
            face_x_resized = round(face_x - face_w * (resize_ratio - 1) / 2)
            face_y_resized = round(face_y - face_h * (resize_ratio - 1) / 2)
            # face_x_resized = round(face_x - face_w * 0.5)
            # face_y_resized = round(face_y - face_h * 0.5)

            face_left = face_x_resized
            face_upper = face_y_resized
            face_right = face_x_resized + face_w_resized
            face_lower = face_y_resized + face_h_resized

            facebox_crop = image.crop((face_left, face_upper, face_right, face_lower))

            cropped_facebox_aspect_ratio = face_w_resized / face_h_resized  # w/h

            if face_w_resized < face_h_resized:
                crop_width = 256
                crop_height = round(256 * (1 / cropped_facebox_aspect_ratio))
                # detecting whether to use face padding
                if image_width < 256 or crop_height > image_height:
                    if image_width < 256 and crop_height > image_height:
                        padding = (0, crop_height - face_h_resized, 256 - face_w_resized, 0)
                        image = expand_image(image, padding)
                    elif image_width < 256:
                        # padding (left up, right, bottom)
                        padding = (0, 0, 256 - face_w_resized, 0)
                        image = expand_image(image, padding)
                    elif image_height < face_h_resized:
                        padding = (0, crop_height - face_h_resized, 0, 0)
                        image = expand_image(image, padding)
            else:
                crop_height = 256
                crop_width = round(256 * cropped_facebox_aspect_ratio)
                if image_height < 256 or crop_width > image_width:
                    if image_height < 256 and crop_width > image_width:
                        padding = (0, 256 - image_height, crop_width - image_width, 0)
                        image = expand_image(image, padding)
                    elif image_height < 256:
                        # padding (left up, right, bottom)
                        padding = (0, 256 - image_height, 0, 0)
                        image = expand_image(image, padding)
                    elif crop_width > image_width:
                        padding = (0, 0, crop_width - image_width, 0)
                        image = expand_image(image, padding)
            facebox_crop_256 = facebox_crop.resize((crop_width, crop_height))

            # Center crop a 224x224 region:
            from torchvision import transforms
            crop_obj = transforms.CenterCrop((224, 224))

            facebox_crop_256_centercrop_224 = crop_obj(facebox_crop_256)

            image_name = row[0][2:]

            crop_image_path = image_save_path + str(image_count) + ".jpg"
            facebox_crop_256_centercrop_224.save(crop_image_path)
            image_count += 1

            # plt.figure("facebox_crop_256_centercrop_224")
            # plt.imshow(facebox_crop_256_centercrop_224)
            # plt.show()

            images.append(facebox_crop_256_centercrop_224)
            paths_cropped.append(crop_image_path)

        # # If MTCNN detects more than 1 face:
        # #   Loop through all the faces that MTCNN detected
        # #     Select the one where MTCNN's bounding_box center is closest to [face_x+(face_w/2), face_y+(face_h/2)]
        # # For the selected bounding box from MTCNN:
        # #    Apply the bounding box transformation for VGGFace2 (Check the paper: It was something like enlargen by 30%, ... do some more stuff, crop, etc.)
        # #       From the VGGFace2 paper: "During training, the extended bounding box of the face is resized so that
        # #       the shorter side is 256 pixels, then a 224×224 pixels region is randomly cropped from each sample.
        # #       The mean value of each channel is subtracted for each pixel."
        # #       "The face bounding box is then extended by a factor of 0.3 to include the whole head"
        # #       "First the extended bounding box of the face is resized so that the shorter side is 256 pixels; then
        # #       the centre 224 × 224 crop of the face image is used as input to the network."
        #
        # # Assuming for now that face_x...etc is the MTCNN bbox:
        # # Let's test this with a non-uniform aspect ratio:
        # face_w = round(face_w * 0.8)  # Don't use this
        # # Draw the original MTCNN facebox:
        # cv2.rectangle(image, (face_x, face_y), (face_x + face_w, face_y + face_h), (255, 0, 0))
        # # Resize the MTCNN facebox by 1.3:
        # face_w_resized = round(face_w * 1.3)
        # face_h_resized = round(face_h * 1.3)
        # face_x_resized = round(face_x - face_w * 0.15)
        # face_y_resized = round(face_y - face_h * 0.15)
        # cv2.rectangle(image, (face_x_resized, face_y_resized), (face_x_resized + face_w_resized, face_y_resized + face_h_resized), (0, 0, 255))
        # cv2.imshow("im", image)
        # #cv2.waitKey()
        #
        # # Crop the enlargened region:
        # # Todo: The enlargened region can be outside the image. Maybe add padding or some other solution, but be
        # #  careful not to mess up the subsequent steps. Maybe also CenterCrop function can be used here?
        # # If you really get stuck with this, you could check whether face_y_resized < 0 etc (whether it's outside the image), and then just skip the image.
        # facebox_crop = image[face_y_resized:face_y_resized + face_h_resized,
        #                face_x_resized:face_x_resized + face_w_resized, :]
        # cv2.imshow("crop", facebox_crop)
        # #cv2.waitKey()
        #
        # # Check which side is the shorter side, and resize that one to 256:
        # cropped_facebox_aspect_ratio = facebox_crop.shape[1] / facebox_crop.shape[0]  # w/h
        # if face_w_resized < face_h_resized:
        #     crop_width = 256
        #     crop_height = round(256 * (1/cropped_facebox_aspect_ratio))
        # else:
        #     crop_height = 256
        #     crop_width = round(256 * cropped_facebox_aspect_ratio)
        # facebox_crop_256 = cv2.resize(facebox_crop, (crop_width, crop_height))
        # cv2.imshow("crop_256", facebox_crop_256)
        #
        # # Center crop a 224x224 region:
        # from torchvision import transforms
        # crop_obj = transforms.CenterCrop((224, 224))
        # import PIL.Image as Image
        # facebox_crop_256_pil = Image.fromarray(facebox_crop_256)
        # facebox_crop_256_centercrop_224 = crop_obj(facebox_crop_256_pil)
        # cv2.imshow("crop_224", np.array(facebox_crop_256_centercrop_224))
        #
        # cv2.waitKey()
        # cv2.destroyWindow("im")
        # cv2.destroyWindow("crop")
        # cv2.destroyWindow("crop_256")
        # cv2.destroyWindow("crop_224")

        #    Run it through VGGFace2 to get the descriptor
        # Store the pair [VGGFace2_descriptor, (yaw, pitch, roll)] to the disk with np.save(...)

        # select_str = "coords.feature_id, coords.x, coords.y"
        # from_str = "featurecoords coords"
        # where_str = "coords.face_id = {}".format(row[1])
        # query_str = "SELECT " + select_str + " FROM " + from_str + " WHERE " + where_str
        # lm = np.zeros((21, 2)).astype(np.float32)
        # v = np.zeros((21)).astype(np.int32)
        #
        # c2 = conn.cursor()
        #
        # for q in c2.execute(query_str):
        #     lm[q[0] - 1][0] = q[1]
        #     lm[q[0] - 1][1] = q[2]
        #     v[q[0] - 1] = 1
        #
        # lm = lm.reshape(42)
        #
        # landmarks.append(lm)
        # visibility.append(v)
        #
        # c2.close()

        # Error correction
        # if(face_x < 0): face_x = 0
        # if(face_y < 0): face_y = 0
        # if(face_w > image_w):
        #   face_w = image_w
        #   face_h = image_w
        # if(face_h > image_h):
        #   face_h = image_h
        #   face_w = image_h

        # Crop the face from the image
        # image_cropped = np.copy(image[face_y:face_y+face_h, face_x:face_x+face_w])
        # Uncomment the lines below if you want to rescale the image to a particular size
        # to_size = 227
        # image_rescaled = cv2.resize(image, (to_size, to_size))
        # Uncomment the line below if you want to use adaptive histogram normalisation
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
        # image_normalised = clahe.apply(image_rescaled)
        # Save the image
        # change "image_cropped" with the last uncommented variable name above
        # cv2.imwrite(output_path, image_cropped)
        # images.append(image_rescaled)

        # Printing the information
        print("Counter: " + str(counter))
        # print("iPath:    " + input_path)
        # print("oPath:    " + output_path)
        # print("x:       " + str(face_x))
        # print("y:       " + str(face_y))
        # print("w:       " + str(face_w))
        # print("h:       " + str(face_h))
        # print("Gender:  " + str(sex))
        # print("Pose:    " + str(np.asarray([roll, pitch, yaw])))
        # print("Image shape: " + str(image_rescaled.shape))
        # print("Face coord shape: " + str(lm.shape))
        # print("Visibility shape: " + str(v.shape))
        # print("")
        # Increasing the counter
        counter = counter + 1
        # if the file does not exits it return an exception

    # Once finished the iteration it closes the database
    c.close()
    print("Images list items: ", len(images))
    print("Pose list items: ", len(pose))
    print("Grey image number: " + str(grey_count))
    return paths, paths_cropped, images, landmarks, pose


def split_data(self, data):
    '''
    Splits data in train and validation sets
    '''
    images, face, landmarks, visibility, pose, gender = data

    x_train, x_test, y_train_face, y_test_face, y_train_landmarks, \
    y_test_landmarks, y_train_visibility, y_test_visibility, \
    y_train_pose, y_test_pose, y_train_gender, y_test_gender = \
        train_test_split(images, face, landmarks, visibility, pose, gender, test_size=0.10, shuffle=True)

    train_data = x_train, y_train_face, y_train_landmarks, y_train_visibility, y_train_pose, y_train_gender
    test_data = x_test, y_test_face, y_test_landmarks, y_test_visibility, y_test_pose, y_test_gender

    return (train_data, test_data)


paths, paths_cropped, images, landmarks, pose = get_raw_data(image_path, database_path)

np.save("D:/AFLW/numpylist2/images_path.npy", paths)
np.save("D:/AFLW/numpylist2/pose_data.npy", pose)
np.savez("D:/AFLW/numpylist2/pose_data_with_name.npz", path=paths_cropped, pose=pose)
# np.save("D:/AFLW/numpylist2/images_data.npy", images)
# np.save("D:/AFLW/numpylist/landmarks_data.npy", landmarks)

# # FIXME: Get histogram for pose [roll, pitch, yaw]
# import matplotlib.pyplot as plt
# import numpy as np
# pose = np.load("D:/AFLW/numpylist2/pose_data.npy")
# plt.xlim(-0.5, 0.5)
# plt.hist(pose[:,0], bins='auto')
# plt.title("Porocessed_roll_distribution")
# plt.savefig("Porocessed_roll_distribution")
# plt.show()
# plt.xlim(-0.5, 0.5)
# plt.hist(pose[:,1], bins='auto')
# plt.title("Porocessed_pitch_distribution")
# plt.savefig("Porocessed_pitch_distribution")
# plt.show()
# plt.xlim(-0.5, 0.5)
# plt.hist(pose[:,2], bins='auto')
# plt.title("Porocessed_yaw_distribution")
# plt.savefig("Porocessed_yaw_distribution")
# plt.show()
