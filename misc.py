import numpy
import random
import argparse
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch import nn
from torchvision import models
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def args():

    parser = argparse.ArgumentParser(description='Main Arguments')
    parser.add_argument('--content_image', type=str)
    parser.add_argument('--style_image', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--content_levels', type=str)
    parser.add_argument('--style_levels', type=str)
    parser.add_argument('--learning_rate', type=int, default = 0.0001)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--style_weight', type=int, default=1000000)
    parser.add_argument('--input_image', type=str, default='random', choices=['clone', 'random'])
    
    args = parser.parse_args()

    return args


def load_device():
    return device

def get_image_size():
    imsize = 512 if torch.cuda.is_available() else 128
    return imsize

def set_seeds(answer_to_everything=42):
    random.seed(answer_to_everything)
    numpy.random.seed(answer_to_everything)
    torch.random.seed(answer_to_everything)


def load_images(content_image_path, style_image_path, input_image, imsize, device):
    content_image = load_image(content_image_path, imsize).to(device, torch.float)
    style_image = load_image(style_image_path, imsize).to(device, torch.float)
    input_image = get_input_image(input_image, content_image, device)
    return content_image, style_image, input_image


def load_image(path, size):
    image = Image.open(path)
    loader = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()]) 
    image = loader(image).unsqueeze(0)
    return image

def get_input_image(input_image_type, content_image, device=None):
    if input_image_type=='random':
        return torch.randn(content_image.data.size(), device=device)
    elif input_image_type=='clone':
        return content_image.clone()




# def imshow(tensor, title=None):
#     image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
#     image = image.squeeze(0)      # remove the fake batch dimension
#     image = transforms.ToPILImage()(image)
#     plt.imshow(image)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001) # pause a bit so that plots are updated

# plt.ion()
# plt.figure()
# imshow(style_img, title='Style Image')
# plt.figure()
# imshow(content_img, title='Content Image')