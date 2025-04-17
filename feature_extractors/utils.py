import subprocess
from pathlib import Path
from moviepy.editor import *
import cv2
import face_recognition

import os
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Load the pre-trained ViT model for image classification
model_name = '{mask}/huggingface/pretrained_model/gender_classification/'
model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

model.eval()

def save_faces_with_gender(frame_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    face_images = []
    frame = face_recognition.load_image_file(frame_path)
    face_locations = face_recognition.face_locations(frame)
    
    for j, (top, right, bottom, left) in enumerate(face_locations):
        face_image = frame[top:bottom, left:right]
        
        # Convert the face image to PIL format
        face_image_pil = Image.fromarray(face_image)
        
        # Preprocess the face image
        inputs = feature_extractor(images=face_image_pil, return_tensors="pt")
        
        # Predict gender
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            gender = 'M' if predicted_class_idx == 0 else 'F'
        
        # Convert the face image back to BGR format as cv2 expects BGR format
        face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

        # Save the face image with gender in the filename
        face_path = os.path.join(output_dir, f"{gender}.jpg")
        cv2.imwrite(face_path, face_image_bgr)
        face_images.append(face_path)
        
        print(f"Face {j}: {gender}")
    
    return face_images

def recognize_faces(frame_path):
    face_locations = []
    frame = face_recognition.load_image_file(frame_path)
    face_locations.append(face_recognition.face_locations(frame))
    return face_locations

def save_faces(frame_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    face_images = []
    frame = face_recognition.load_image_file(frame_path)
    face_locations=face_recognition.face_locations(frame)
    for j, (top, right, bottom, left) in enumerate(face_locations):
        face_image = frame[top:bottom, left:right]
        
        face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        face_path = os.path.join(output_dir, f"face_{j}.jpg")
        cv2.imwrite(face_path, face_image_bgr)
        face_images.append(face_path)
    return face_images

def extract_audio_from_mp4(mp4_path, audio_output_path):
    video = VideoFileClip(mp4_path, audio_fps=16000, audio_nbytes=2)
    audio = video.audio
    audio.write_audiofile(audio_output_path, codec='pcm_s16le')
    audio.close()
    video.close()

def ffmpeg_extract(in_file, out_path, mode='audio', fps : int = 10) -> None:
    """
    Extract audio/image from input video file and save to disk.

    Args:
        in_file: Path to the input file, e.g. mp4 file.
        out_path: Path to the output file.
        mode: Should be 'audio' or 'image'.
        fps: Frames per second, will be ignored if mode is 'audio'.

    """
    assert mode in ['audio', 'image'], "Parameter 'mode' must be 'audio' or 'image'."
    
    if mode == 'audio':
        args = ['ffmpeg', '-i', in_file, '-vn', '-acodec', 'pcm_s16le', '-y', out_path]
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError("ffmpeg", out, err)
        
    elif mode == 'image':
        # Extract the duration of the video using ffprobe
        duration_command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', in_file]
        duration_process = subprocess.Popen(duration_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        duration_output, duration_err = duration_process.communicate()
        
        # Debugging output
        print(f"Duration command output: {duration_output}")
        print(f"Duration command error: {duration_err}")
        
        if duration_process.returncode != 0:
            raise RuntimeError("ffmpeg", duration_output, duration_err)
        
        duration = float(duration_output.strip())
        midpoint = duration / 2

        # Extract the middle frame
        args = ['ffmpeg', '-i', in_file, '-vf', f'select=gte(t\,{midpoint})', '-frames:v', '1', '-y', out_path]
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        
        # Debugging output
        print(f"Extract frame command output: {out}")
        print(f"Extract frame command error: {err}")
        
        if p.returncode != 0:
            raise RuntimeError("ffmpeg", out, err)


    # elif mode == 'image':
    #     # args = ['ffmpeg', '-i', in_file, '-vf', f'fps={fps}', '-y', str(Path(out_path) / '%03d.bmp')]
    #     # 提取最关键的帧（通常是第一个 I 帧）
    #     # args = ['ffmpeg', '-i', in_file, '-vf', 'select=eq(pict_type\,I)', '-vsync', 'vfr', '-frames:v', '1', '-y', str(out_path)]
    #     # p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     # out, err = p.communicate()
    #     # if p.returncode != 0:
    #     #     raise RuntimeError("ffmpeg", out, err)
        
    #     # Extract the duration of the video
    #     duration_command = ['ffmpeg', '-i', in_file, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0']
    #     duration_process = subprocess.Popen(duration_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     duration_output, duration_err = duration_process.communicate()
    #     if duration_process.returncode != 0:
    #         raise RuntimeError("ffmpeg", duration_output, duration_err)
        
    #     duration = float(duration_output.strip())
    #     midpoint = duration / 2

    #     # Extract the middle frame
    #     args = ['ffmpeg', '-i', in_file, '-vf', f'select=gte(t\,{midpoint})', '-frames:v', '1', '-y', out_path]
    #     p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     out, err = p.communicate()
    #     if p.returncode != 0:
    #         raise RuntimeError("ffmpeg", out, err)
