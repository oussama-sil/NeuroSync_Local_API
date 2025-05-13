
# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import json
import pprint
import time
from flask import request, jsonify
import librosa
import numpy as np
import torch
import flask

from utils.audio.extraction.extract_features import extract_and_combine_features
from utils.audio.processing.audio_processing import process_audio_features
from utils.generate_face_shapes import generate_facial_data_from_bytes
from utils.model.dimension_scalars import scale_blendshapes_by_section
from utils.model.faceblendshape import FaceBlendShape
from utils.model.model import load_model
from utils.config import config

app = flask.Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Activated device:", device)

model_path = 'utils/model/model.pth'
blendshape_model = load_model(model_path, config, device)



audio_path = '/home/osilem/ressources/audio/15.mp3'

# Count inference time 

start_time = time.time()


# Load audio file 
y, sr = librosa.load(audio_path, sr=88200)
max_val = np.max(np.abs(y))
if max_val > 0:
    y = y / max_val

print(sr)


# Infor about audio 
frame_length = int(0.01667 * sr)  # Frame length set to 0.01667 seconds (~60 fps)
hop_length = frame_length // 2  # 2x overlap for smoother transitions
min_frames = 9  # Minimum number of frames needed for delta calculation
num_frames = (len(y) - frame_length) // hop_length + 1
if num_frames < min_frames:
    print(f"Audio file is too short: {num_frames} frames, required: {min_frames} frames")
    exit(-1)
    
combined_features = extract_and_combine_features(y, sr, frame_length, hop_length)
blendshapes = process_audio_features(combined_features, blendshape_model, device, config)

facial_data = []
for frame in blendshapes:
    frame_data = [float(value) for value in frame]
    facial_data.append(frame_data)


# print(torch.tensor(facial_data)[2])
print(torch.tensor(facial_data)[10:15,52:55])

# Scaling blendshapes 
scaled_facial_data = [scale_blendshapes_by_section(
            blendshape, 
            1.0, 
            1.0, 
            0.6, 
            eyewide_left_scale=0.4, 
            eyewide_right_scale=0.4, 
            eyesquint_left_scale=1.0, 
            eyesquint_right_scale=1.0
        ) for blendshape in facial_data]

print(torch.tensor(scaled_facial_data)[0:15,52:55])


# Mapping Values to blendshapes names  

frames = [
    {name.name[0].lower() + name.name[1:]: blendshape[name.value] for name in FaceBlendShape}
for blendshape in scaled_facial_data
]

# pprint.pprint([frame['headRoll'] for frame in frames])

end_time = time.time()
print(f"Inference time: {end_time - start_time} seconds")



#  Formatting for Blender Rendering 
file_path = '/mnt/d/ressources/animation/15_neurosync.json'
with open(file_path, 'w') as json_file:
    json.dump(frames, json_file, indent=4)



