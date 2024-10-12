import modal
import os
import boto3

# Define the Modal image
from modal import Image, Stub, wsgi_app, Secret, Volume,Mount,web_endpoint
from PIL import Image as PILImage, ImageOps
import numpy as np

stub = Stub("anydoor")
vol = Volume.from_name("generation_store")
aws_credentials = {
    'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
    'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
    'region_name': os.getenv('AWS_REGION')  # Optional
}

session = boto3.Session(
aws_access_key_id=aws_credentials['aws_access_key_id'],
aws_secret_access_key=aws_credentials['aws_secret_access_key'],
aws_session_token=aws_credentials.get('aws_session_token')
)


image = (
    Image.debian_slim()
    .apt_install("libglib2.0-0", "libglib2.0-dev", "libsm6", "libxext6", "libxrender-dev", "libgtk2.0-dev")
    .pip_install('opencv_python==4.5.5.64')
    .pip_install("opencv_python_headless==4.5.5.64")
    .pip_install("albumentations==1.3.0")
    .pip_install("einops==0.3.0")
    .pip_install("fvcore==0.1.5.post20221221")
    .pip_install("gradio==3.39.0")
    .pip_install("numpy==1.23.1")

    .pip_install("omegaconf==2.1.1")
    .pip_install("open_clip_torch==2.17.1")
    .pip_install("boto3==1.34.146")
    .pip_install("botocore==1.34.146")
    .pip_install("Pillow==9.4.0")
    .pip_install("pytorch_lightning==1.5.0")
    .pip_install("safetensors==0.2.7")
    .pip_install("scipy==1.9.1")
    .pip_install("setuptools==66.0.0")
    .pip_install("share==1.0.4")
    .pip_install("submitit==1.5.1")
    .pip_install("timm==0.6.12")
    .pip_install("torchmetrics==0.6.0")
    .pip_install("torch==2.0.0+cu117", extra_index_url="https://download.pytorch.org/whl/cu117")
    .pip_install("torchvision==0.15.0", extra_index_url="https://download.pytorch.org/whl/cu117")
    .pip_install("tqdm==4.65.0")
    .pip_install("transformers==4.19.2")
    .pip_install("xformers==0.0.18")
)
local_directory_path = "/Users/adelabdalla/Desktop/FashionSwap/target_input"  # Change this to your actual local path

@stub.function(
    image=image, 
    gpu="A10G",
    container_idle_timeout=300,
    mounts=[
        Mount.from_local_dir("/Users/adelabdalla/Desktop/FashionSwap/AnyDoor", remote_path="/FashionSwap/AnyDoor"),
        Mount.from_local_dir("/Users/adelabdalla/Desktop/FashionSwap/user_input", remote_path="/FashionSwap/user_input"),
        Mount.from_local_dir("/Users/adelabdalla/Desktop/FashionSwap/target_input", remote_path="/FashionSwap/target_input"),
        Mount.from_local_dir("/Users/adelabdalla/Desktop/FashionSwap/segments", remote_path="/FashionSwap/segments")
    ],
    volumes={"/generations": vol}
)
@web_endpoint(method="POST")
def my_function(input_image: str, input_mask: str, reference_image: str, reference_mask: str = None):
    '''
    The input image is the image of you that you will upload to us to try on clothes. 
    The reference image is the item of clothing you want to try on. 
    For the MVP (since we're not deploying anything) we're going to save the url of the item you want to wear
    locally then this function will take it and use it in here. 

    The file paths should be relative ones, so that we can add /FashionSwap/AnyDoor to them.
    '''
    import sys
    import numpy as np
    from PIL import Image as PILImage
    import cv2
    import requests
    from io import BytesIO

    sys.path.append('/FashionSwap/AnyDoor')  # Add AnyDoor to the Python path
    sys.path.append('/FashionSwap/AnyDoor/dinov2')

    from run_inference import inference_single_image

    # Process input paths
    input_list = [input_image, input_mask]
    for index_path in range(len(input_list)):
        if input_list[index_path] is not None:
            input_list[index_path] = '/FashionSwap' + '/' + input_list[index_path]

    input_image = input_list[0]
    input_mask = input_list[1]

    # Download and process the reference image from URL
    response = requests.get(reference_image)
    ref_image = np.array(PILImage.open(BytesIO(response.content))).astype(np.uint8)
    if ref_image.shape[-1] == 4:  # Check if image has an alpha channel
        ref_mask = (ref_image[:, :, -1] > 128).astype(np.uint8)
        ref_image = ref_image[:, :, :3]  # Remove alpha channel
    else:
        ref_mask = np.ones(ref_image.shape[:2], dtype=np.uint8)  # Default mask if no alpha channel

    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR)

    # Load and process the input image and mask
    back_image = cv2.imread(input_image).astype(np.uint8)
    back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

    tar_mask = cv2.imread(input_mask, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    tar_mask = (tar_mask > 128).astype(np.uint8)

    # Perform inference
    gen_image = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask)

    # Save the generated image
    save_path = "/generations/generated_image.png"
    # Create an S3 client using the credentials
    is_success, buffer = cv2.imencode('.png', gen_image[:, :, ::-1])
    print(is_success)
    image_data = BytesIO(buffer).getvalue()


# Use the session to create a client
    s3_client = session.client('s3')

    s3_client.put_object(Bucket='fashion-swap', Key=f'{save_path}', Body=image_data, ContentType='image/png')
    

    print(f"Image saved at {save_path}")
    return {"url": "https://fashion-swap.s3.us-east-2.amazonaws.com//generations/generated_image.png"}



# #Example function call

# if __name__ == "__main__":
#     with stub.run():
#         my_function.remote('user_input/08909_00.jpg','segments/input_cloth_components/1.png','https://imgs.search.brave.com/l0crkXKPwUKBczZWu5fPDBadf6jA2aIXkG2onZiBQys/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tLm1l/ZGlhLWFtYXpvbi5j/b20vaW1hZ2VzL0kv/ODE5TkhjOVhZd0wu/anBn')
