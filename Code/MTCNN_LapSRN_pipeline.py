import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import time
import cv2
import mediapipe as mp
import torch
import yt_dlp
from RealESRGAN import RealESRGAN
from mtcnn import MTCNN
from numpy import asarray
from PIL import Image
from cv2 import dnn_superres
import numpy as np

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('success!')

detector = MTCNN()

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.8)


def sharpen_image(img, alpha=.7):
    kernel = np.array([[0, -1, 0],
                       [-1, 5 + alpha, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(img, -1, kernel)
    return sharpened_image


def upscale_with_real_lapsrn(img_array, lapsrn_model):
    sr_model = dnn_superres.DnnSuperResImpl_create()
    sr_model.readModel(lapsrn_model)
    sr_model.setModel('lapsrn', 4)
    sr_image_array = sr_model.upsample(img_array)
    return sr_image_array


def get_video_stream_url(video_url):
    ydl_opts = {
        'format': 'best',
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        video_url = info_dict.get('url', None)
    return video_url


def process_video(video_path, output_path, lapsrn_model,process_every=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Original FPS: {fps}, Width: {width}, Height: {height}")

    original_image_path = "/home/aro/Schreibtisch/virtual_python/SuperResolution/Group Study/Output/Images/margin/original"  #Adjust the parent directory as needed
    upsampled_image_path = "/home/aro/Schreibtisch/virtual_python/SuperResolution/Group Study/Output/Images/margin/upsampled" #Adjust the parent directory as needed

    # Ensure directories exist
    os.makedirs(original_image_path, exist_ok=True)
    os.makedirs(upsampled_image_path, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    detection_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % process_every == 0:
            detections = detector.detect_faces(frame)

            if detections:
                for face in detections:

                    detection_count += 1
                    ih, iw, _ = frame.shape
                    x, y, w, h = face['box']

                    original_crop = frame[y:y + h, x:x + w]
                    print(f'Bounding Box of original crop  of {detection_count}: x={x}, y={y}, w={w}, h={h}')
                    cv2.imwrite(os.path.join(original_image_path, f'original_crop_{detection_count}.png'), original_crop)

                    x_start = max(0, x - 5)
                    y_start = max(0, y - 5)
                    x_end = min(frame.shape[1], x + w + 5)
                    y_end = min(frame.shape[0], y + h + 5)

                    cropped_face_with_margin = frame[y_start:y_end, x_start:x_end]

                    upsampled_face = upscale_with_real_lapsrn(cropped_face_with_margin, lapsrn_model)

                    sharpened_face = sharpen_image(upsampled_face)

                    if face_detection.process(sharpened_face):
                        print('Redetection is successful!')

                        # Position the upsampled face exactly where the original face was without reducing its size
                        new_x = max(0, x_start)
                        new_y = max(0, y_start)

                        '''
                        the variable 'sharpened_face' is the result of the upsampling after confirmation by redetection.
                        '''

                        # Determine the size of the upsmapled face
                        new_w = min(sharpened_face.shape[1], frame.shape[1] - new_x)
                        new_h = min(sharpened_face.shape[0], frame.shape[0] - new_y)

                        # Place the upscaled face in the image
                        frame[new_y:new_y + new_h, new_x:new_x + new_w] = sharpened_face[:new_h, :new_w]
                        cv2.imwrite(os.path.join(upsampled_image_path,f'upscaled_detection_{detection_count}.png'), sharpened_face)



                        # Draw the bounding box
                        cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
                        cv2.putText(frame, 'REDETECTION', (300, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=3,
                                    color=(0, 0, 0))

                        # Write the processed frame to the video
                        out.write(frame)
                    else:
                        print('Redetection not possible')

                    print(
                        f'Bounding Box after upscaling of {detection_count}: x={new_x}, y={new_y}, w={new_w}, h={new_h}')

        cv2.imshow('MediaPipe Face Detection', frame)

       # if detection_count >= 30:
            #break

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video processing with face detection and upscaling.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video.")
    parser.add_argument('--output_path', type=str, default='output_friday.mp4', help="Path to save the processed video.")
    parser.add_argument('--lapsrn_model', type=str, required=True, help="Path to the LAPSRN model.")
    #parser.add_argument('--margin', type=int, default=20, help="Margin around detected faces for cropping.")
    parser.add_argument('--process_every', type=int, default=1, help="Process every Nth frame.")

    args = parser.parse_args()

    process_video(args.video_path, args.output_path, args.lapsrn_model, args.process_every) #, args.margin
