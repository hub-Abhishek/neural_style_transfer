import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch import nn
from torchvision import models
from torchsummary import summary

import misc
import nn_misc




import torch.optim as optim
def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    # optimizer = optim.LBFGS([input_img])
    optimizer = optim.Adam([input_img])
    return optimizer


num_steps = 350
model, content_losses, style_losses = get_model_and_losses(cnn, content_layers, style_layers, content_img, style_img)
input_img.requires_grad_(True)
model.requires_grad_(False)
optimizer = get_input_optimizer(input_img)
print('Optimizing..')
run = [0]
while run[0] <= num_steps:
    style_weight=1000000
    content_weight=1
    # run[0] += 1
    
    def closure():
        # correct the values of updated input image
        with torch.no_grad():
            input_img.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss

        style_score *= style_weight
        content_score *= content_weight

        loss = style_score + content_score
        loss.backward()

        
        if run[0] % 50 == 0:
            print("run {}:".format(run))
            print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                style_score.item(), content_score.item()))
            print()

        return style_score + content_score
    optimizer.step(closure)
    run[0] += 1
    print(run[0])
with torch.no_grad():
    input_img.clamp_(0, 1)


if __name__=="__main__":
    args = misc.args()

    content_image_path = "./dancing.jpg" # args.content_image
    style_image_path = "./picasso.jpg" # args.style_image
    model_name = args.model_name
    content_levels = args.content_levels
    style_levels = args.style_levels
    learning_rate = args.learning_rate
    epochs = args.epochs
    style_weight = args.style_weight
    input_image = args.input_image
    
    misc.set_seeds(42)

    device = misc.load_device()
    imsize = misc.get_image_size()
    content_image, style_image, input_image = misc.load_images(content_image_path, style_image_path, input_image, imsize, device)

    assert content_image.size() == style_image.size()

    model = nn_misc.load_base_model(model_name, device)

    