#import align.detect_face
#import tensorflow as tf
#import argparse
#https://github.com/xirikm/MTCNN-PyTorch

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#
# pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
#
# minsize = 20  # minimum size of face
# threshold = [0.6, 0.7, 0.7]  # three steps's threshold
# factor = 0.709  # scale factor


from mtcnn import FaceDetector
from PIL import Image
import cv2
import numpy as np
# 人脸检测对象。优先使用GPU进行计算（会自动判断GPU是否可用）
# 你也可以通过设置 FaceDetector("cpu") 或者 FaceDetector("cuda") 手动指定计算设备
detector = FaceDetector()

image = Image.open(r"C:\Users\Mingrui\Desktop\drawed_image.jpg")
#image = Image.open(r"D:\AFLW\aflw\data\flickr\3\image00143.jpg")


# image = cv2.imread(r"D:\AFLW\aflw\data\flickr\3\image00902.jpg",0)
# # 然后用ctvcolor（）函数，进行图像变换。
# image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# image = np.array(image)
# image = Image.fromarray(image)



# 检测人脸，返回人脸位置坐标
# 其中bboxes是一个n*5的列表、landmarks是一个n*10的列表，n表示检测出来的人脸个数，数据详细情况如下：
# bbox：[左上角x坐标, 左上角y坐标, 右下角x坐标, 右下角y坐标, 检测评分]
# landmark：[右眼x, 左眼x, 鼻子x, 右嘴角x, 左嘴角x, 右眼y, 左眼y, 鼻子y, 右嘴角y, 左嘴角y]
bboxes, landmarks = detector.detect(image)

# 绘制并保存标注图
drawed_image = detector.draw_bboxes(image,factor = 0)
drawed_image.save("./drawed_image.jpg")

# 裁剪人脸图片并保存
face_img_list = detector.crop_faces(image, size=256)
for i in range(len(face_img_list)):
    face_img_list[i].save("./face_" + str(i + 1) + ".jpg")

bb_weight_hight =[]
for b in bboxes:
    box_w = b[2] - b[0]
    box_h = b[3] - b[1]
    bb_weight_hight.append((box_w,box_h))


def detect_bboxes(bboxes):
    new_bboxes = []
    for b in bboxes:
        box_w = b[2] - b[0]
        box_h = b[3] - b[1]

        if box_w or box_h > 100:
            new_bboxes.append(b)
    return new_bboxes

def get_face_count(image_path):
    image = Image.open(image_path)

    bboxes, _ = detector.detect(image)
    bboxes = detect_bboxes(bboxes)
    return len(bboxes)


