import cv2 
from model import rcnn
from util import preprocess_img, postprocess_tens
import streamlit as st 

colorizer_rcnn = rcnn(pretrained=True).eval()
use_gpu = False
if(use_gpu):
    colorizer_rcnn.cuda()
    
st.title('Video Colorization')

def main():
    
    up_vdo = st.file_uploader('Upload a Video', type=['mp4'])
    if up_vdo is not None:
        with open('./Videos/input.mp4', 'wb') as f:
            f.write(up_vdo.getvalue())
            
    if st.button('Colorize'):
        cap = cv2.VideoCapture('./Videos/input.mp4')
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Resized_Window", 800, 600)


        while True:
            ret, img = cap.read()
            if not ret:
                st.warning('Completed Successfully')
                break
            
            (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
            
            output_img = postprocess_tens(tens_l_orig, colorizer_rcnn(tens_l_rs).cpu())

            cv2.imshow("Resized_Window", output_img)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            
            
            
if __name__=='__main__':
    main()