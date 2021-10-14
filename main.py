from __future__ import division
from keras_csp.resnet50_full_res import nn_p3p4p5_fpn as network

import os
import cv2
import time
from keras.layers import Input, MaxPooling2D
from keras.models import Model, Sequential
from keras_csp import config, bbox_process
from keras_csp.utilsfunc import *

from PIL import Image
from tools.utils import hex_to_rgb
import streamlit as st
from io import BytesIO
import base64
import os


def model_load(offset, num_scale):
    with st.spinner('Getting the Neurons in Order ...'):
        img_input = Input(shape=(None, None, 3))
        preds = network(img_input, offset=offset, num_scale=num_scale, trainable=False)
        model = Model(img_input, preds)
        nms_max = Sequential()
        nms_max.add(MaxPooling2D(pool_size=(3, 3), strides=1, padding='same', input_shape=(None, None, 1)))
        model.load_weights("data/models/net_e32_l0.0814104549587.hdf5", by_name=True)
    return model, nms_max


# general
st.set_page_config(layout="wide")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# download model
model_path = "data/models/net_e32_l0.0814104549587.hdf5"
if not os.path.exists(model_path):
    with st.spinner('Downloading Model ...'):
        os.system("bash data/models/download_model.sh")

# get model
C = config.Config()
C.offset, C.scale, C.num_scale, C.down, C.size_test = False, 'h', 1, 1, [0, 0]
model, nms_max = model_load(C.offset, C.num_scale)


def draw_bbox(img, num_dets, dets, rgb_color, text_color, bg_color, thr=.41):

    image = img.copy()

    text_thickness = 2
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75

    bboxes = np.array(dets, dtype='float32')
    bboxes = bboxes[bboxes[:, 4] > thr]
    bboxes = np.array([[x[0], x[1], x[2] - x[0] + 1, x[3] - x[1] + 1, x[4]] for x in bboxes])

    for i in range(bboxes.shape[0]):
        x1, y1 = int(bboxes[i, 0]), int(bboxes[i, 1])
        x2, y2 = x1 + int(bboxes[i, 2]), y1 + int(bboxes[i, 3])
        cv2.rectangle(image, (x1, y1), (x2, y2), rgb_color, 2)

    # add the text
    text = str(int(num_dets))
    size = cv2.getTextSize(text, font_face, font_scale, text_thickness)
    x, y = 50, 50
    cv2.rectangle(image, (x, y - size[0][1] - size[1]), (x + size[0][0], y + size[0][1] - size[1]), bg_color, cv2.FILLED)
    cv2.putText(image, text, (x, y), font_face, font_scale, text_color, text_thickness)

    return image


def run(img, rgb_color, text_color, bg_color):

    # network stuff
    score = 0.2
    soft_box_thre = 0.3
    limit = 3.5

    # grab filepaths for our images from
    start_time = time.time()

    def detect_face(img, scale=1, flip=False):
        img_h, img_w = img.shape[:2]
        img_h_new, img_w_new = int(np.ceil(scale * img_h / 16) * 16), int(np.ceil(scale * img_w / 16) * 16)
        scale_h, scale_w = img_h_new / img_h, img_w_new / img_w

        img_s = cv2.resize(img, None, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)
        # print frame_number
        C.size_test[0] = img_h_new
        C.size_test[1] = img_w_new

        if flip:
            img_sf = cv2.flip(img_s, 1)
            x_rcnn = format_img(img_sf, C)
        else:
            x_rcnn = format_img(img_s, C)
        Y = model.predict(x_rcnn)
        Y_max = nms_max.predict(Y[0])
        keep = (Y_max == Y[0])
        Y[0] = Y[0] * keep
        if C.offset:
            boxes = bbox_process.parse_shanghai_h_offset(Y, C, score=score, down=C.down, nmsthre=0.4)
        else:
            boxes = bbox_process.parse_shanghai_h_nooff(Y, C, score=score, down=C.down, nmsthre=0.4)
        if len(boxes) > 0:
            keep_index = np.where(np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) >= limit)[0]
            boxes = boxes[keep_index, :]
        if len(boxes) > 0:
            if flip:
                boxes[:, [0, 2]] = img_s.shape[1] - boxes[:, [2, 0]]
            boxes[:, 0:4:2] = boxes[:, 0:4:2] / scale_w
            boxes[:, 1:4:2] = boxes[:, 1:4:2] / scale_h
        else:
            boxes = np.empty(shape=[0, 5], dtype=np.float32)
        return boxes

    def im_det_ms_pyramid(image, max_im_shrink):
        # shrink detecting and shrink only detect big face
        det_s = np.row_stack((detect_face(image, 0.5), detect_face(image, 0.5, flip=True)))
        index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 64)[0]
        det_s = det_s[index, :]

        det_temp = np.row_stack((detect_face(image, 0.75), detect_face(image, 0.75, flip=True)))
        index = np.where(np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) > 42)[0]
        det_temp = det_temp[index, :]
        det_s = np.row_stack((det_s, det_temp))

        det_temp = np.row_stack((detect_face(image, 0.25), detect_face(image, 0.25, flip=True)))
        index = np.where(np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) > 128)[0]
        det_temp = det_temp[index, :]
        det_s = np.row_stack((det_s, det_temp))

        st = [1.5, 2.0]
        for i in range(len(st)):
            if (st[i] <= max_im_shrink):
                det_temp = np.row_stack((detect_face(image, st[i]), detect_face(image, st[i], flip=True)))

                if st[i] == 1.5:
                    index = np.where(
                        np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 12)[0]
                    det_temp = det_temp[index, :]

                elif st[i] == 2.0:
                    index = np.where(
                        np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 8)[0]
                    det_temp = det_temp[index, :]

                det_s = np.row_stack((det_s, det_temp))
        return det_s

    max_im_shrink = (0x7fffffff / 577.0 / (img.shape[0] * img.shape[1])) ** 0.5  # the max size of input image
    shrink = max_im_shrink if max_im_shrink < 1 else 1

    det0 = detect_face(img)
    det1 = detect_face(img, flip=True)
    det2 = im_det_ms_pyramid(img, max_im_shrink)

    # merge all test results via bounding box voting
    det = np.row_stack((det0, det1, det2))
    keep_index = np.where(np.minimum(det[:, 2] - det[:, 0], det[:, 3] - det[:, 1]) >= 2)[0]  # >= 3
    dets = det[keep_index, :]
    dets = bbox_process.soft_bbox_vote(dets, thre=soft_box_thre, score=score)

    keep_index = np.where((dets[:, 2] - dets[:, 0] + 1) * (dets[:, 3] - dets[:, 1] + 1) >= 2 ** 2)[0]
    dets = dets[keep_index, :]

    # send dets to draw_bbox
    output_image = draw_bbox(img, len(dets), dets, rgb_color, text_color, bg_color)
    print(time.time() - start_time)

    return output_image


def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


def main():

    # create homepage feel
    st.image(os.path.join('refs', 'tokyo_ppl.jpeg'), use_column_width=True)
    st.markdown("<h1 style='text-align: center; color: white;'>Point Supervised Crowd Detection</h1>", unsafe_allow_html=True)

    # project descriptions
    st.markdown("This project is a point-based crowd detection system for the ShanghaiTech dataset. \
                The dataset contains over 1 million people from 100 different groups. The dataset is used to \
                train a point-based crowd detection system made in Keras. The system is based on the paper \
                <a href='https://ieeexplore.ieee.org/abstract/document/9347744'> A Self-Training Approach for \
                Point-Supervised Object Detection and Counting in Crowds</a>. Adapted from  \
                <a href='https://github.com/WangyiNTU/Point-supervised-crowd-detection'> Crowd-SDNet: \
                Point-supervised crowd self-detection network</a>. Check it out for yourself below!", unsafe_allow_html=True)

    # make columns for st
    col1, col2, col3, col4 = st.columns(4)
    fps = col1.slider('FPS Output for Video', 1, 20, 5, 1)
    color = col2.color_picker('Bounding Box Color', '#FF0900')
    text_color = col3.color_picker('Count Text Color', '#000000')
    bg_color = col4.color_picker('Count Background Color', '#ffffff')

    # menu
    upload = st.file_uploader('Upload Images/Video', type=['jpg', 'jpeg', 'png', 'gif', 'avi', 'mp4'])

    if upload is not None:
        filetype = upload.name.split('.')[-1]
        rgb_color, text_color, bg_color = hex_to_rgb(color), hex_to_rgb(text_color), hex_to_rgb(bg_color)

        if filetype in ['jpg', 'jpeg', 'png']:
            image_bytes = Image.open(upload).convert('RGB')
            image = np.array(image_bytes)  # if you want to pass it to OpenCV
            with st.spinner('Detecting and Counting ...'):
                output_image = run(image, rgb_color, text_color, bg_color)
            output_image_bytes = Image.fromarray(np.uint8(output_image))
            st.image(output_image_bytes, use_column_width=True)
            st.markdown(get_image_download_link(output_image_bytes, 'crowd_counter.png', 'Download'), unsafe_allow_html=True)

        elif filetype in ['avi', 'mp4', 'gif']:
            st.info('Video is not supported yet.')
            st.video(upload)


if __name__ == '__main__':
    main()
