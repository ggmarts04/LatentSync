import sys
import subprocess
import runpod
import os
import tempfile
import requests
import uuid
from predict import Predictor # Assuming Predictor is in predict.py

# Global predictor instance to reuse (RunPod might keep workers warm)
# Setup will be called if the instance is not ready
predictor_instance = None

def download_file(url, destination_dir):
    """Downloads a file from a URL to a specified directory using pget."""
    print(f"Downloading {url} to {destination_dir}")
    # Ensure pget is available or handle potential errors
    try:
        # Using pget for potentially faster downloads, similar to cog.yaml
        # pget automatically extracts if it's a tar archive, we might need to adjust if it's just a raw file
        # For simple file download, ensure pget doesn't try to extract.
        # The -x flag in cog.yaml is for extraction. We might not need it if downloading raw video/audio.
        # However, pget might be smart enough. If not, might need to use curl or requests.
        # Let's assume pget handles direct file downloads correctly or we can adjust command.
        # Create a unique filename within the destination directory
        filename = os.path.join(destination_dir, os.path.basename(url))
        subprocess.check_call(["pget", url, filename], close_fds=False) # Simpler pget command for single file
        print(f"Successfully downloaded {url} to {filename}")
        return filename
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {e}")
        raise
    except FileNotFoundError:
        print("Error: pget command not found. Ensure it is installed and in PATH.")
        # Fallback or error
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

    # Parameters for the model, with defaults matching predict.py if not provided
    guidance_scale = float(job_input.get('guidance_scale', 2.0))
    inference_steps = int(job_input.get('inference_steps', 20))
    seed = int(job_input.get('seed', 0)) # 0 for random seed as per predict.py

    # Initialize predictor if not already done
    if predictor_instance is None:
        print("Initializing Predictor...")
        predictor_instance = Predictor()
        predictor_instance.setup() # This downloads model weights, sets up links
        print("Predictor initialized.")

    try:
        # Create temporary directories for downloaded files
        with tempfile.TemporaryDirectory() as tmpdir_video, tempfile.TemporaryDirectory() as tmpdir_audio:
            print(f"Created temporary directories: {tmpdir_video}, {tmpdir_audio}")
            
            local_video_path = download_file(video_url, tmpdir_video)
            local_audio_path = download_file(audio_url, tmpdir_audio)

            print(f"Calling predictor.predict with video: {local_video_path}, audio: {local_audio_path}")
            # Call the predict method
            output_path_object = predictor_instance.predict(
                video=local_video_path, # predict.py expects string paths
                audio=local_audio_path, # predict.py expects string paths
                guidance_scale=guidance_scale,
                inference_steps=inference_steps,
                seed=seed
            )
            
            output_path_str = str(output_path_object) # Convert Path object to string
            print(f"Prediction successful. Output at: {output_path_str}")

            # --- Start BunnyCDN Upload Logic ---
            bunny_access_key = os.environ.get("BUNNY_ACCESS_KEY")
            if not bunny_access_key:
                print("HANDLER: Error - BUNNY_ACCESS_KEY environment variable not set.")
                return {"error": "BunnyCDN Access Key not configured on server."}

            local_file_path = output_path_str # This is '/tmp/video_out.mp4'
            
            # Generate a unique filename for cloud storage to prevent overwrites
            unique_filename = f"{uuid.uuid4()}.mp4"
            
            bunny_storage_url = f"https://storage.bunnycdn.com/zockto/videos/{unique_filename}"
            public_download_url = f"https://zockto.b-cdn.net/videos/{unique_filename}"

            headers = {
                "AccessKey": bunny_access_key,
                "Content-Type": "video/mp4"
            }

            try:
                print(f"HANDLER: Uploading {local_file_path} to BunnyCDN at {bunny_storage_url}...")
                with open(local_file_path, 'rb') as f:
                    video_data = f.read()
                
                response = requests.put(bunny_storage_url, headers=headers, data=video_data)
                
                if response.status_code == 201:
                    print(f"HANDLER: Successfully uploaded to BunnyCDN. Public URL: {public_download_url}")
                    # Replace the old return statement with this one
                    # The original 'print("HANDLER: Processing complete, returning output.")' will be replaced by the one below
                    print("HANDLER: Processing and upload complete, returning public URL.")
                    return {"output_url": public_download_url}
                else:
                    error_message = f"HANDLER: Error uploading to BunnyCDN. Status: {response.status_code}, Response: {response.text}"
                    print(error_message)
                    return {"error": "Failed to upload video to BunnyCDN.", "details": response.text}

            except FileNotFoundError:
                print(f"HANDLER: Error - Local video file not found at {local_file_path} for upload.")
                return {"error": "Output video file not found for upload."}
            except Exception as upload_e:
                print(f"HANDLER: An unexpected error occurred during BunnyCDN upload: {str(upload_e)}")
                return {"error": f"An unexpected error occurred during video upload: {str(upload_e)}"}
            # --- End BunnyCDN Upload Logic ---

    except Exception as e:
        print(f"Error during prediction: {e}")
        print(f"HANDLER: Exception caught ({e}), returning error.")
        return {"error": str(e)}

if __name__ == '__main__':
    # This part is for local testing of the handler, if needed.
    # It simulates a RunPod event.
    # You would need to have pget installed and model weights accessible.
    # And ensure predict.py and its dependencies are in PYTHONPATH.
    
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
