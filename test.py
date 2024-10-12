import boto3
from io import BytesIO
import cv2

# Example AWS credentials stored in a dictionary
aws_credentials = {
    'aws_access_key_id': 'AKIAVFIFEQFHVJHL2SPX',
    'aws_secret_access_key': 'e+RN58+QOhqtRrh3jg0ccd0r1aeCZ8oygMJ7j/cE',
    'region_name': 'us-east-2'  # Optional
}

# Create a Boto3 session using the credentials
session = boto3.Session(
    aws_access_key_id=aws_credentials['aws_access_key_id'],
    aws_secret_access_key=aws_credentials['aws_secret_access_key'],
    aws_session_token=aws_credentials.get('aws_session_token'),
    region_name=aws_credentials.get('region_name')
)

# Create an S3 client using the session
s3_client = session.client('s3')

# Assume gen_image is your generated image array
# For example, this line simulates loading an image from a path
save_path = "user_input/08909_00.jpg"
gen_image = cv2.imread(save_path)  # Load an image to simulate gen_image

# Encode the image to a buffer
is_success, buffer = cv2.imencode('.png', gen_image[:, :, ::-1])

if is_success:
    image_data = BytesIO(buffer).getvalue()
    # Upload the image to S3
    s3_client.put_object(Bucket='fashion-swap', Key=f'{save_path}', Body=image_data,    ContentType='image/png')
    print(f"Image uploaded to S3 at s3://fashion-swap/{save_path}")
else:
    print("Failed to encode the image")
