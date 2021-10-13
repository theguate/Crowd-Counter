from __future__ import division
import os
import sys
import cv2
import time
import argparse

sys.path.insert(0, '../')
sys.path.insert(0, './')
sys.path.insert(0, '.')


def plot(input_images_dir, input_detections_dir, output_dir):

    thr = 0.41
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # -- display utils
    text_color = (90, 90, 90)
    text_thickness = 2
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75

    # file paths
    filepaths = [os.path.join(input_images_dir, file).replace("\\", "/") for file in os.listdir(input_images_dir)]
    num_imgs = len(filepaths)
    start_time = time.time()

    print('{} Images being Processed to {} '.format(num_imgs, output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for f in range(num_imgs):
        filepath = filepaths[f]
        filename = filepath.split('/')[-1].split('.')[0]
        jpgpath = os.path.join(output_dir, filename + '.png')

        img = cv2.imread(filepath)

        file_detections_path = input_detections_dir
        txtpath = os.path.join(file_detections_path, filename + '.txt')
        with open(txtpath, 'rb') as fid:
            lines = fid.readlines()

        bboxes = []
        for line in lines[2:]:
            bboxes.append([float(i) for i in line.split()])
        bboxes = np.array(bboxes, dtype='float32')
        bboxes = bboxes[bboxes[:, 4] > thr]

        for i in range(bboxes.shape[0]):
            x1, y1 = int(bboxes[i, 0]), int(bboxes[i, 1])
            x2, y2 = x1 + int(bboxes[i, 2]), y1 + int(bboxes[i, 3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # add the text
        text = str(int(lines[1]))
        size = cv2.getTextSize(text, font_face, font_scale, text_thickness)
        x, y = 50, 50
        cv2.rectangle(img, (x, y - size[0][1] - size[1]),
                      (x + size[0][0], y + size[0][1] - size[1]), (255, 255, 255),
                      cv2.FILLED)
        cv2.putText(img, text, (x, y), font_face, font_scale, text_color,
                    text_thickness)

        cv2.imwrite(jpgpath, img)

    print(time.time() - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot ShanghaiTech detections')
    parser.add_argument('--input_images_dir', type=str, default='data/images', help='input images dir')
    parser.add_argument('--input_detections_dir', type=str, default='output/detection', help='input detections dir')
    parser.add_argument('--output_dir', type=str, default='output/detection_view', help='output dir')
    args = parser.parse_args()
    plot(args.input_images_dir, args.input_detections_dir, args.output_dir)
