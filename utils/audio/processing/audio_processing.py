# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

# audio_processing.py

import numpy as np
import torch
from torch.cuda.amp import autocast

def decode_audio_chunk(audio_chunk, model, device, config):
    # Use precision based on config
    use_half_precision = config.get("use_half_precision", True)
    
    # Force float16 if half precision is desired; else float32
    dtype = torch.float16 if use_half_precision else torch.float32

    # Convert audio chunk directly to the desired precision
    src_tensor = torch.tensor(audio_chunk, dtype=dtype).unsqueeze(0).to(device)

    with torch.no_grad():
        if use_half_precision:

            with autocast(dtype=torch.float16):
                encoder_outputs = model.encoder(src_tensor)
                output_sequence = model.decoder(encoder_outputs)
        else:
            encoder_outputs = model.encoder(src_tensor)
            output_sequence = model.decoder(encoder_outputs)

        # Convert output tensor back to numpy array
        decoded_outputs = output_sequence.squeeze(0).cpu().numpy()
    return decoded_outputs


def concatenate_outputs(all_decoded_outputs, num_frames):
    final_decoded_outputs = np.concatenate(all_decoded_outputs, axis=0)
    final_decoded_outputs = final_decoded_outputs[:num_frames]
    return final_decoded_outputs

def ensure_2d(final_decoded_outputs):
    if final_decoded_outputs.ndim == 3:
        final_decoded_outputs = final_decoded_outputs.reshape(-1, final_decoded_outputs.shape[-1])
    return final_decoded_outputs

def pad_audio_chunk(audio_chunk, frame_length, num_features, pad_mode='replicate'):
    """
    Pads the audio_chunk to ensure it has a number of frames equal to frame_length.
    
    Parameters:
        audio_chunk (np.array): Input audio data with shape (num_frames, num_features).
        frame_length (int): Desired number of frames.
        num_features (int): Number of features per frame.
        pad_mode (str): Type of padding to use. Options are:
                        - 'reflect': Pads using reflection.
                        - 'replicate': Pads by replicating the last frame.
    
    Returns:
        np.array: Padded audio_chunk with shape (frame_length, num_features).
    """
    if audio_chunk.shape[0] < frame_length:
        pad_length = frame_length - audio_chunk.shape[0]
        
        if pad_mode == 'reflect':
            # --- Original reflect padding method ---
            padding = np.pad(
                audio_chunk,
                pad_width=((0, pad_length), (0, 0)),
                mode='reflect'
            )
            # Using the last pad_length frames from the reflected padding
            audio_chunk = np.vstack((audio_chunk, padding[-pad_length:, :num_features]))
        
        elif pad_mode == 'replicate':
            # --- New replicate padding method ---
            # Replicate the last frame to fill the remaining frames
            last_frame = audio_chunk[-1:]  # Select the last frame (shape: (1, num_features))
            replication = np.tile(last_frame, (pad_length, 1))  # Replicate it pad_length times
            audio_chunk = np.vstack((audio_chunk, replication))
        
        else:
            raise ValueError(f"Unsupported pad_mode: {pad_mode}. Choose 'reflect' or 'replicate'.")
    
    return audio_chunk


def blend_chunks(chunk1, chunk2, overlap):
    actual_overlap = min(overlap, len(chunk1), len(chunk2))
    if actual_overlap == 0:
        return np.vstack((chunk1, chunk2))
    
    blended_chunk = np.copy(chunk1)
    for i in range(actual_overlap):
        alpha = i / actual_overlap 
        blended_chunk[-actual_overlap + i] = (1 - alpha) * chunk1[-actual_overlap + i] + alpha * chunk2[i]
        
    return np.vstack((blended_chunk, chunk2[actual_overlap:]))

def process_audio_features(audio_features, model, device, config):
    # Configuration settings
    frame_length = config['frame_size']  # Number of frames per chunk (e.g., 64)
    overlap = config.get('overlap', 32)  # Number of overlapping frames between chunks
    num_features = audio_features.shape[1]
    num_frames = audio_features.shape[0]
    all_decoded_outputs = []

    # Set model to evaluation mode
    model.eval()

    # Process chunks with the specified overlap
    start_idx = 0
    while start_idx < num_frames:
        end_idx = min(start_idx + frame_length, num_frames)

        # Select and pad chunk if needed
        audio_chunk = audio_features[start_idx:end_idx]
        audio_chunk = pad_audio_chunk(audio_chunk, frame_length, num_features)

        # 🔥 Pass config to dynamically choose precision
        decoded_outputs = decode_audio_chunk(audio_chunk, model, device, config)
        decoded_outputs = decoded_outputs[:end_idx - start_idx]

        # Blend with the last chunk if it exists
        if all_decoded_outputs:
            last_chunk = all_decoded_outputs.pop()
            blended_chunk = blend_chunks(last_chunk, decoded_outputs, overlap)
            all_decoded_outputs.append(blended_chunk)
        else:
            all_decoded_outputs.append(decoded_outputs)

        # Move start index forward by (frame_length - overlap)
        start_idx += frame_length - overlap

    # Process any remaining frames to ensure total frame count matches input
    current_length = sum(len(chunk) for chunk in all_decoded_outputs)
    if current_length < num_frames:
        remaining_frames = num_frames - current_length
        final_chunk_start = num_frames - remaining_frames
        audio_chunk = audio_features[final_chunk_start:num_frames]
        audio_chunk = pad_audio_chunk(audio_chunk, frame_length, num_features)
        decoded_outputs = decode_audio_chunk(audio_chunk, model, device, config)
        all_decoded_outputs.append(decoded_outputs[:remaining_frames])

    # Concatenate all chunks and trim to the original frame count
    final_decoded_outputs = np.concatenate(all_decoded_outputs, axis=0)[:num_frames]

    # Normalize or apply any post-processing
    final_decoded_outputs = ensure_2d(final_decoded_outputs)
    final_decoded_outputs[:, :61] /= 100  # Normalize specific columns

    # Easing effect for smooth start (fades in first 0.2 seconds)
    ease_duration_frames = min(int(0.1 * 60), final_decoded_outputs.shape[0])
    easing_factors = np.linspace(0, 1, ease_duration_frames)[:, None]
    final_decoded_outputs[:ease_duration_frames] *= easing_factors

    # Zero out unnecessary columns (optional post-processing)
    final_decoded_outputs = zero_columns(final_decoded_outputs)

    return final_decoded_outputs


def zero_columns(data):
    # Old columns to zero out
    # columns_to_zero = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    columns_to_zero = [ 55, 56, 57, 58, 59, 60]
    modified_data = np.copy(data) 
    modified_data[:, columns_to_zero] = 0
    return modified_data
