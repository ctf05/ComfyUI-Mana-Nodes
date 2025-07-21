import os
import torch
import hashlib
import folder_paths
from ..helpers.utils import pil2tensor
from PIL import Image
from pathlib import Path

class video2audio:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        try:
            input_dir = os.path.join(folder_paths.get_input_directory(), "video")
            os.makedirs(input_dir, exist_ok=True)
            
            # Limit the number of files to prevent freezing with large directories
            all_files = os.listdir(input_dir)
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv')
            files = []
            
            for f in all_files[:100]:  # Limit to first 100 files
                if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(video_extensions):
                    files.append(f"video/{f}")
            
            if not files:
                files = ["No videos found"]
        except Exception as e:
            print(f"Error loading video files: {e}")
            files = ["Error loading videos"]
        
        return {
            "required": {
                "video": (sorted(files), {"mana_video_upload": True}),
                "frame_limit": ("INT", {"default": 16, "min": 1, "max": 10240, "step": 1}),
                "frame_start": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1}),
                "filename_prefix": ("STRING", {"default": "audio\\audio"})
            },
            "optional": {}
        }

    CATEGORY = "💠 Mana Nodes"
    RETURN_TYPES = ("IMAGE", "STRING","INT", "INT", "INT","INT",) 
    RETURN_NAMES = ("images", "audio_file","fps","frame_count", "height", "width",)
    FUNCTION = "run"

    def run(self, **kwargs):
        video_path = folder_paths.get_annotated_filepath(kwargs['video'])
        
        # Extract frames
        frames, width, height = self.extract_frames(video_path, kwargs)
        
        # Extract audio
        video_path_obj = Path(video_path)
        audio, fps = self.extract_audio_with_moviepy(video_path_obj, kwargs)
        
        if not frames:
            raise ValueError("No frames could be extracted from the video.")
        if not audio:
            audio = "No audio track in the video."
            
        return (torch.cat(frames, dim=0), audio, fps, len(frames), height, width,)
    
    def extract_audio_with_moviepy(self, video_path, kwargs):
        # Lazy import moviepy only when needed
        try:
            from moviepy.editor import VideoFileClip
        except ImportError as e:
            print("MoviePy not installed. Please install it with: pip install moviepy")
            return None, 30  # Return default fps
        except Exception as e:
            print(f"Error importing moviepy: {e}")
            return None, 30
        
        # Convert WindowsPath object to string
        video_file_path_str = str(video_path)
        
        try:
            # Load the video file
            video = VideoFileClip(video_file_path_str)
            
            # Check if the video has an audio track
            if video.audio is None:
                video.close()
                return None, video.fps
            
            # Calculate start and end time in seconds
            fps = video.fps  # frames per second
            start_time = kwargs['frame_start'] / fps
            end_time = (kwargs['frame_start'] + kwargs['frame_limit']) / fps
            
            # Ensure end_time doesn't exceed video duration
            end_time = min(end_time, video.duration)
            
            full_path = os.path.join(folder_paths.get_output_directory(), os.path.normpath(kwargs['filename_prefix']))
            if not full_path.endswith('.wav'):
                full_path += '.wav'
            Path(os.path.dirname(full_path)).mkdir(parents=True, exist_ok=True)
            full_path_to_audio = os.path.abspath(full_path)
            
            # Extract the specific audio segment
            audio = video.subclip(start_time, end_time).audio
            audio.write_audiofile(full_path, logger=None)  # Disable moviepy's verbose logging
            
            # Clean up
            audio.close()
            video.close()
            
            return full_path_to_audio, fps
            
        except Exception as e:
            print(f"Error extracting audio with moviepy: {e}")
            if 'video' in locals():
                video.close()
            return None, 30  # Return default fps

    def extract_frames(self, video_path, kwargs):
        # Lazy import and ensure opencv only when needed
        from ..helpers.utils import ensure_opencv
        ensure_opencv()
        
        try:
            import cv2
        except ImportError as e:
            print("OpenCV not installed. Please install it with: pip install opencv-python-headless")
            return [], 0, 0
        
        try:
            video = cv2.VideoCapture(video_path)
            
            if not video.isOpened():
                print(f"Failed to open video file: {video_path}")
                return [], 0, 0
            
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Validate frame_start
            if kwargs['frame_start'] >= total_frames:
                print(f"frame_start ({kwargs['frame_start']}) exceeds total frames ({total_frames})")
                video.release()
                return [], width, height
            
            video.set(cv2.CAP_PROP_POS_FRAMES, kwargs['frame_start'])
            
            frames = []
            frames_to_read = min(kwargs['frame_limit'], total_frames - kwargs['frame_start'])
            
            for i in range(frames_to_read):
                ret, frame = video.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(pil2tensor(Image.fromarray(frame)))
                else:
                    break
            
            video.release()
            return frames, width, height
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            if 'video' in locals():
                video.release()
            return [], 0, 0

    @classmethod
    def IS_CHANGED(cls, video, *args, **kwargs):
        if video == "No videos found" or video == "Error loading videos":
            return ""
        
        try:
            video_path = folder_paths.get_annotated_filepath(video)
            m = hashlib.sha256()
            with open(video_path, "rb") as f:
                # Read in chunks to avoid memory issues with large files
                for chunk in iter(lambda: f.read(4096), b""):
                    m.update(chunk)
            return m.digest().hex()
        except Exception as e:
            print(f"Error in IS_CHANGED: {e}")
            return ""

    @classmethod
    def VALIDATE_INPUTS(cls, video, *args, **kwargs):
        if video == "No videos found" or video == "Error loading videos":
            return f"Invalid video file: {video}"
        
        try:
            if not folder_paths.exists_annotated_filepath(video):
                return f"Invalid video file: {video}"
            return True
        except Exception as e:
            return f"Error validating video: {str(e)}"
