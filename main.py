import os
import torch
import gdown
import warnings
import numpy as np
from PIL import Image
import streamlit as st
from model import SASNet
from utils import pred_map_overlay, get_image_download_link, transform_images, hex_to_rgb, dotdict

# setttings
args = dotdict({'block_size': 32, 'batch_size': 4, 'log_para': 1000, 'model_path': "models/SHHA.pth"})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings('ignore')
st.set_page_config()


@st.cache(show_spinner=False)
def load_model(args):
    if not os.path.exists(args.model_path):
        with st.spinner('Downloading Model ...'):
            gdown.download("https://drive.google.com/uc?id={}".format("1d4oOXdWr0e0tp47Fz9Jl7xu5vAvyz7BF"), output=args.model_path)
    with st.spinner('Getting Neruons in Order ...'):
        model = SASNet(args=args)
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        return model


def run(img, rgb_color, text_color, bg_color):

    # get model/data
    model = load_model(args)
    image = transform_images(img)

    # run
    with torch.no_grad():
        model.eval()
        pred = model(image)
        pred_map = pred.cpu().detach().numpy()
        pred_count = np.sum(pred_map[0]) / args.log_para

    # draw bbox
    output_image = pred_map_overlay(pred, pred_count, rgb_color, text_color, bg_color)

    return output_image


def main():

    # create homepage feel
    st.image(os.path.join('refs', 'tokyo_ppl.jpeg'), use_column_width=True)
    st.markdown("<h1 style='text-align: center; color: white;'>CrowdCounting-SASNet</h1>", unsafe_allow_html=True)

    # project descriptions
    st.markdown("This project is an implmentation of the CrowdCounting-SASNet model trained on the ShanghaiTech A dataset. \
                The Shanghaitech dataset is a large-scale crowd counting dataset. It consists of 1198 annotated crowd images. \
                In total, the dataset consists of 330,165 annotated people. Images from Part-A were collected from the Internet, \
                while images from Part-B were collected on the busy streets of Shanghai. \
                The model is based on the paper \
                <a href='https://www.aaai.org/AAAI21Papers/AAAI-6452.SongQ.pdf'> To Choose or to Fuse? Scale Selection for Crowd Counting</a>. Adapted from  \
                <a href='https://github.com/TencentYoutuResearch/CrowdCounting-SASNet'>CrowdCounting-SASNet</a>.  \
                Check it out for yourself below!", unsafe_allow_html=True)

    # make columns for st
    col1, col2, col3 = st.columns(3)
    fps = col1.slider('FPS Output for Video', 1, 20, 5, 1)
    text_color = col2.color_picker('Count Text Color', '#000000')
    bg_color = col3.color_picker('Count Background Color', '#ffffff')

    # menu
    upload = st.file_uploader('Upload Images/Video', type=['jpg', 'jpeg', 'png', 'gif', 'avi', 'mp4'])

    if upload is not None:
        filetype = upload.name.split('.')[-1]
        rgb_color, text_color, bg_color = None, hex_to_rgb(text_color), hex_to_rgb(bg_color)

        if filetype in ['jpg', 'jpeg', 'png']:
            with st.spinner('Detecting and Counting ...'):
                image_bytes = Image.open(upload).convert('RGB')
                image = np.array(image_bytes)
                output_image = run(image, rgb_color, text_color, bg_color)
                output_image_bytes = Image.fromarray(np.uint8(output_image))
            st.image(output_image_bytes, use_column_width=True)
            st.markdown(get_image_download_link(output_image_bytes, 'crowd_counter.png', 'Download'), unsafe_allow_html=True)

        elif filetype in ['avi', 'mp4', 'gif']:
            st.info('Video is not supported yet.')
            st.video(upload)

        else:
            st.warning('Unsupported file type.')


if __name__ == '__main__':
    model = load_model(args)
    main()
