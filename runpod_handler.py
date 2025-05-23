import subprocess
import os
import tempfile
from predict import Predictor  # Assuming Predictor is in predict.py

# Global predictor instance to reuse (RunPod might keep workers warm)
predictor_instance = None

def download_file(url, destination_dir):
    """Downloads a file from a URL to a specified directory using pget."""
    print(f"Downloading {url} to {destination_dir}")
    try:
        filename = os.path.join(destination_dir, os.path.basename(url))
        subprocess.check_call(["pget", url, filename], close_fds=False)
        print(f"Successfully downloaded {url} to {filename}")
        return filename
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {e}")
        raise
    except FileNotFoundError:
        print("Error: pget command not found. Ensure it is installed and in PATH.")
        raise Exception("pget not found. Cannot download files.")

def handler(event):
    global predictor_instance

    job_input = event.get('input', {})
    if not job_input:
        return {"error": "No input provided in event"}

    video_url = job_input.get('video_url')
    audio_url = job_input.get('audio_url')

    if not video_url or not audio_url:
        return {"error": "video_url and audio_url are required."}

    guidance_scale = float(job_input.get('guidance_scale', 2.0))
    inference_steps = int(job_input.get('inference_steps', 20))
    seed = int(job_input.get('seed', 0))  # 0 means random seed

    if predictor_instance is None:
        print("Initializing Predictor...")
        predictor_instance = Predictor()
        predictor_instance.setup()
        print("Predictor initialized.")

    try:
        with tempfile.TemporaryDirectory() as tmpdir_video, tempfile.TemporaryDirectory() as tmpdir_audio:
            print(f"Created temporary directories: {tmpdir_video}, {tmpdir_audio}")

            local_video_path = download_file(video_url, tmpdir_video)
            local_audio_path = download_file(audio_url, tmpdir_audio)

            print(f"Calling predictor.predict with video: {local_video_path}, audio: {local_audio_path}")
            output_path_object = predictor_instance.predict(
                video=local_video_path,
                audio=local_audio_path,
                guidance_scale=guidance_scale,
                inference_steps=inference_steps,
                seed=seed
            )

            output_path_str = str(output_path_object)
            print(f"Prediction successful. Output at: {output_path_str}")
            return {"output_path": output_path_str}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}
