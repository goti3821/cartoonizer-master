import os
import io
import uuid
import sys
import yaml
import traceback
import cv2
import numpy as np
import skvideo.io
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory

# Load configuration options
with open('./config.yaml', 'r') as fd:
    opts = yaml.safe_load(fd)

# Update the system path
sys.path.insert(0, './white_box_cartoonizer/')

# Import cartoonizer
from cartoonize import WB_Cartoonize

# Initialize the cartoonizer and load its weights
wb_cartoonizer = WB_Cartoonize(os.path.abspath("white_box_cartoonizer/saved_models/"), opts['gpu'])

# Initialize Flask app
app = Flask(__name__)

# Create folders to store processed images and videos
os.makedirs('static/cartoonized_images', exist_ok=True)
os.makedirs('static/uploaded_videos', exist_ok=True)

def convert_bytes_to_image(img_bytes):
    """Convert bytes to numpy array."""
    pil_image = Image.open(io.BytesIO(img_bytes))
    if pil_image.mode == "RGBA":
        image = Image.new("RGB", pil_image.size, (255, 255, 255))
        image.paste(pil_image, mask=pil_image.split()[3])
    else:
        image = pil_image.convert('RGB')
    
    image = np.array(image)
    return image

def cartoonize_image(img_bytes):
    """Cartoonize an image given as bytes."""
    try:
        image = convert_bytes_to_image(img_bytes)
        img_name = str(uuid.uuid4())
        cartoon_image = wb_cartoonizer.infer(image)
        cartoonized_img_name = os.path.join('static/cartoonized_images', img_name + ".jpg")
        cv2.imwrite(cartoonized_img_name, cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))
        return cartoonized_img_name
    except Exception as e:
        print(f"Error during image cartoonization: {e}")
        traceback.print_exc()
        return None

def process_video(video_path):
    """Process a video for cartoonization."""
    try:
        filename = str(uuid.uuid4()) + ".mp4"
        original_video_path = video_path
        modified_video_path = os.path.join('static/uploaded_videos', filename.split(".")[0] + "_modified.mp4")

        # Fetch metadata and set frame rate
        file_metadata = skvideo.io.ffprobe(original_video_path)
        original_frame_rate = file_metadata.get('video', {}).get('@r_frame_rate', '30/1')
        output_frame_rate = opts['output_frame_rate'] if not opts['original_frame_rate'] else original_frame_rate
        output_frame_rate_number = int(output_frame_rate.split('/')[0])

        width_resize = opts['resize-dim']

        # Resize and convert video
        if opts['trim-video']:
            time_limit = opts['trim-video-length']
            os.system(f"ffmpeg -hide_banner -loglevel warning -ss 0 -i '{original_video_path}' -t {time_limit} -filter:v scale={width_resize}:-2 -r {output_frame_rate_number} -c:a copy '{modified_video_path}'")
        else:
            os.system(f"ffmpeg -hide_banner -loglevel warning -ss 0 -i '{original_video_path}' -filter:v scale={width_resize}:-2 -r {output_frame_rate_number} -c:a copy '{modified_video_path}'")

        audio_file_path = os.path.join('static/uploaded_videos', filename.split(".")[0] + "_audio_modified.mp4")
        os.system(f"ffmpeg -hide_banner -loglevel warning -i '{modified_video_path}' -map 0:1 -vn -acodec copy -strict -2 '{audio_file_path}'")

        cartoon_video_path = wb_cartoonizer.process_video(modified_video_path, output_frame_rate)

        # Add audio to the cartoonized video
        final_cartoon_video_path = os.path.join('static/uploaded_videos', filename.split(".")[0] + "_cartoon_audio.mp4")
        os.system(f"ffmpeg -hide_banner -loglevel warning -i '{cartoon_video_path}' -i '{audio_file_path}' -codec copy -shortest '{final_cartoon_video_path}'")

        # Clean up
        os.remove(original_video_path)
        os.remove(modified_video_path)
        os.remove(audio_file_path)
        os.remove(cartoon_video_path)

        return final_cartoon_video_path
    except Exception as e:
        print(f"Error during video processing: {e}")
        traceback.print_exc()
        return None

# API to cartoonize an image
@app.route('/cartoonize_image', methods=['POST'])
def cartoonize_image_api():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    img_file = request.files['image']
    img_bytes = img_file.read()

    cartoonized_image_path = cartoonize_image(img_bytes)
    if cartoonized_image_path:
        return jsonify({"cartoonized_image": cartoonized_image_path})
    else:
        return jsonify({"error": "Failed to cartoonize image"}), 500

# API to cartoonize a video
@app.route('/cartoonize_video', methods=['POST'])
def cartoonize_video_api():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    video_path = os.path.join('static/uploaded_videos', video_file.filename)
    video_file.save(video_path)

    cartoonized_video_path = process_video(video_path)
    if cartoonized_video_path:
        return jsonify({"cartoonized_video": cartoonized_video_path})
    else:
        return jsonify({"error": "Failed to cartoonize video"}), 500

# Serve cartoonized images and videos
@app.route('/static/<path:filename>', methods=['GET'])
def serve_file(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(debug=True)
