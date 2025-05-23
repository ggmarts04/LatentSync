# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import subprocess

MODEL_CACHE = "checkpoints"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download the model weights
        print("Downloading model files from Hugging Face Hub (ByteDance/LatentSync-1.5)...")
        model_cache_dir = "checkpoints" # This is MODEL_CACHE
        try:
            subprocess.check_call([
                "huggingface-cli", "download", "ByteDance/LatentSync-1.5",
                "--local-dir", model_cache_dir,
                "--local-dir-use-symlinks", "False", # Ensure actual files are downloaded
                "--exclude", "*.git*", "*.md" # Exclude .git files/folders and markdown files
            ], close_fds=False) # Mimic close_fds from previous pget call
            print("Model files downloaded successfully using huggingface-cli.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading model files using huggingface-cli: {e}")
            raise # Re-raise the exception to stop the setup if download fails
        except FileNotFoundError:
            print("Error: huggingface-cli command not found. Ensure it is installed and in PATH.")
            raise # Re-raise the exception

        # Soft links for the auxiliary models
        os.system("mkdir -p ~/.cache/torch/hub/checkpoints")
        
        link_path_1 = os.path.expanduser("~/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip")
        if not os.path.exists(link_path_1):
            os.system(
                "ln -s $(pwd)/checkpoints/auxiliary/2DFAN4-cd938726ad.zip ~/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip"
            )
        
        link_path_2 = os.path.expanduser("~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth")
        if not os.path.exists(link_path_2):
            os.system(
                "ln -s $(pwd)/checkpoints/auxiliary/s3fd-619a316812.pth ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth"
            )
            
        link_path_3 = os.path.expanduser("~/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
        if not os.path.exists(link_path_3):
            os.system(
                "ln -s $(pwd)/checkpoints/auxiliary/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth"
            )

    def predict(
        self,
        video: Path = Input(description="Input video", default=None),
        audio: Path = Input(description="Input audio to ", default=None),
        guidance_scale: float = Input(description="Guidance scale", ge=1, le=3, default=2.0),
        inference_steps: int = Input(description="Inference steps", ge=20, le=50, default=20),
        seed: int = Input(description="Set to 0 for Random seed", default=0),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed <= 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        video_path = str(video)
        audio_path = str(audio)
        config_path = "configs/unet/stage2.yaml"
        ckpt_path = "checkpoints/latentsync_unet.pt"
        output_path = "/tmp/video_out.mp4"

        # Run the following command:
        command = [
            "python", "-m", "scripts.inference",
            "--unet_config_path", config_path,
            "--inference_ckpt_path", ckpt_path,
            "--guidance_scale", str(guidance_scale),
            "--video_path", video_path,
            "--audio_path", audio_path,
            "--video_out_path", output_path,
            "--seed", str(seed),
            "--inference_steps", str(inference_steps)
        ]
        subprocess.check_call(command)
        return Path(output_path)
