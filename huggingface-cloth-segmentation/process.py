from network import U2NET
import shutil

import os
from PIL import Image
import cv2
import gdown
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from collections import OrderedDict
from options import opt


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


class Normalize_image(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"




def apply_transform(img):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    return transform_rgb(img)



def generate_mask(input_image, first_upload,net, palette, device = 'cpu'):
    '''
    This function generates the segmentation of clothing 
    Use case is a bool representing initial upload. 
    Initial upload here refers to us segmenting the user's clothing rather than the online clothing. 
    In other words when a user signs up for the first time he/she will upload their image of themselves. 

    This will be the case most of the time as the reference images (the one's that you want to try on) do not need any masks. 
    '''

    #img = Image.open(input_image).convert('RGB')
    img = input_image
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = apply_transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    if first_upload:
        alpha_out_dir = os.path.join(opt.output,'input_components')
        cloth_seg_out_dir = os.path.join(opt.output,'input_cloth_segmentation')
    else:


        alpha_out_dir = os.path.join(opt.output,'reference_components')
        cloth_seg_out_dir = os.path.join(opt.output,'reference_cloth_segmentation_total')
        # Clear out the directory if it exists
        if os.path.exists(alpha_out_dir):
            shutil.rmtree(alpha_out_dir)
        if os.path.exists(cloth_seg_out_dir):
            shutil.rmtree(cloth_seg_out_dir)
        
        # Create new empty directories
        os.makedirs(alpha_out_dir, exist_ok=True)
        os.makedirs(cloth_seg_out_dir, exist_ok=True)

    os.makedirs(alpha_out_dir, exist_ok=True)
    os.makedirs(cloth_seg_out_dir, exist_ok=True)

    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    classes_to_save = []

    # Check which classes are present in the image
    for cls in range(1, 4):  # Exclude background class (0)
        if np.any(output_arr == cls):
            classes_to_save.append(cls)

    # Save alpha masks
    for cls in classes_to_save:
        alpha_mask = (output_arr == cls).astype(np.uint8) * 255
        alpha_mask = alpha_mask[0]  # Selecting the first channel to make it 2D
        alpha_mask_img = Image.fromarray(alpha_mask, mode='L')
        alpha_mask_img = alpha_mask_img.resize(img_size, Image.BICUBIC)
        alpha_mask_img.save(os.path.join(alpha_out_dir, f'{cls}.png'))

    # Save final cloth segmentations
    cloth_seg = Image.fromarray(output_arr[0].astype(np.uint8), mode='P')
    cloth_seg.putpalette(palette)
    cloth_seg = cloth_seg.resize(img_size, Image.BICUBIC)
    cloth_seg.save(os.path.join(cloth_seg_out_dir, 'final_seg.png'))
    return cloth_seg



def check_or_download_model(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        url = "https://drive.google.com/uc?id=11xTBALOeUkyuaK3l60CpkYHLTmv7k3dY"
        gdown.download(url, file_path, quiet=False)
        print("Model downloaded successfully.")
    else:
        print("Model already exists.")


def load_seg_model(checkpoint_path='/model/cloth_segm.pth', device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)
    if not checkpoint_path:
        check_or_download_model(checkpoint_path)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    return net


def main(image, checkpoint_path, first_upload, cuda):

    device = 'cuda:0' if cuda else 'cpu'

    # Create an instance of your model
    model = load_seg_model(checkpoint_path, device=device)

    palette = get_palette(4)

    cloth_seg = generate_mask(image,first_upload, net=model, palette=palette, device=device)
    return cloth_seg



def segment(image_path, first_upload, checkpoint_path='model/cloth_segm.pth', use_cuda=False):
    """
    This segments the image that is given in the image_path.
    The first image outputted in the cloth_components folder gives you the segmented top. 
    The second image outputted in the cloth_components folder gives you the segmented bottom. 
    If you want both you can look into the cloth_segmentation_total folder


    We segment twice. 
    1. When a user uploads an image of themselves, we segment. 
    2. When a user presses an item that he/she likes we segment. 

    The first case is stored under the 'initial' folders

    Args:
    image_path (str): Path to the input image.
    checkpoint_path (str): Path to the checkpoint file.
    use_cuda (bool): Flag to use CUDA if available.
    """
    img = Image.open(image_path).convert('RGB')
    cloth_seg = main(img, checkpoint_path,first_upload,use_cuda)
    return cloth_seg


#segment('input/08909_00.jpg',False)