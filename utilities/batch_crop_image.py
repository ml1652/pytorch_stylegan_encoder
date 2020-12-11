import cv2 as cv
import os


def crop_image(image_dir, output_path):   # image_dir 批量处理图像文件夹 size 裁剪后的尺寸
    # 获取图片路径列表
    file_path_list = []
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        file_path_list.append(file_path)

    # 逐张读取图片剪裁
    for counter, image_path in enumerate(file_path_list):
        image = cv.imread(image_path)
        image_h = int(image.shape[0])
        image_w = int(image.shape[1])
        if image_h > image_w:
            image_h_new = round(256*(image_h/image_w))
            image = cv.resize(image, (256,image_h_new), interpolation=cv.INTER_AREA)
            cv.imwrite(output_path + "img_resize" + str(counter) + ".jpg",
                       image)
        else:
            image_w_new = round(256*(image_w / image_h))
            image = cv.resize(image, (image_w_new,256), interpolation=cv.INTER_AREA)
            cv.imwrite(output_path + "img_resize" + str(counter) + ".jpg",
                       image)

        image_h_resize = int(image.shape[0])
        image_w_resize = int(image.shape[1])
        # image = cv.resize(image, (224,224), interpolation = cv.INTER_AREA)
        # center_crop_size = 224
        # mid_x, mid_y = int(size / 2), int(size / 2)
        # mid_y = mid_y +16
        # cw2, ch2 = int(center_crop_size / 2), int(center_crop_size / 2)
        # img_crop = image[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        #
        # cv.imwrite(output_path + "img_" + str(counter) + ".jpg",
        #            img_crop)

        center_crop_size = 224
        mid_x, mid_y = int(image_w_resize / 2), int(image_h_resize / 2)
        #mid_y = mid_y +16
        cw2, ch2 = int(center_crop_size / 2), int(center_crop_size / 2)
        img_crop = image[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]

        cv.imwrite(output_path + "img_" + str(counter) + ".jpg",
                   img_crop)

    # #center crop
    # center_crop_size = 224
    # mid_x, mid_y = int(crop_size_w / 2), int(crop_size_h / 2)
    # mid_y = mid_y +16
    # cw2, ch2 = int(center_crop_size / 2), int(center_crop_size / 2)
    # img_crop = img_crop[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]



if __name__ == "__main__":
    image_dir = r"C:\Users\Mingrui\Desktop\datasets\n007241\\"
    output_path = r"C:\Users\Mingrui\Desktop\datasets\n007241_crop\\"
    crop_image(image_dir, output_path)