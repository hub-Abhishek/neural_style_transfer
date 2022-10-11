import misc

import torch
from torch import nn
from torchvision import models

imagenet_mean =  torch.tensor([0.485, 0.456, 0.406]).to(misc.device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).to(misc.device)

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

class normalize(nn.Module):
    def __init__(self, mean=None, std=None):
        super(normalize, self).__init__()
        self.mean = imagenet_mean if mean is None else mean 
        self.std = imagenet_std if std is None else std
    
    def forward(self, x):
        return (x-self.mean)/self.std

class contentLoss(nn.Module):
    def __init__(self, target):
        super(contentLoss, self).__init__()
        self.target = target.detach()
    
    def forward(self, x):
        # calculate content loss between current input and target
        pass


class styleLoss(nn.Module):
    def __init__(self, target):
        super(styleLoss, self).__init__()
        self.target = self.gram_matrix(target.detach())
    
    def forward(self, x):
        # calculate style loss between current values and target
        pass

    @staticmethod
    def gram_matrix(x):
        # calculate gram matrix of x
        pass

def load_base_model(model_name, device):
    if 'vgg' in model_name.lower():
        model = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1', progress=True).features.to(device).eval()
        return model
    else:
        print("couldn't find model!")


def get_model(content_image, style_image, model_name, device):
    base_model = load_base_model(model_name, device)
    model = nn.Sequential()

    normalize_layer = normalize()
    model.add_module('normalize', normalize_layer)

    content_losses = []
    style_losses = []
    i=0

    for layer in base_model:
        if isinstance(layer, nn.Conv2d):
            i+=1
            name = f'conv_{i}'
        elif isinstance(layer, nn.MaxPool2d):
            name = f'max_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
        else:
            print('Unknown layer')
            return 
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_image).detach()
            content_loss = contentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)
        
        if name in style_layers:
            target = model(style_image).detach()
            style_loss = styleLoss(target)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)
        
    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], contentLoss) or isinstance(model[i], styleLoss):
            model = model[:i+1]
            break
    
    return model, content_losses, style_losses
        







# class contentLoss(nn.Module):
#     def __init__(self, target):
#         super(contentLoss, self).__init__()
#         self.target = target.detach()
#     def forward(self, input):
#         self.loss = F.mse_loss(input, self.target)
#         self.save = input
#         return input


# class styleLoss(nn.Module):
#     def __init__(self, target, i):
#         super(styleLoss, self).__init__()
#         self.target = self.gram_matrix(target).detach()
    
#     @staticmethod
#     def gram_matrix(input):
#         a, b, c, d = input.size()
#         features = input.view(a * b, c * d)
#         G = torch.mm(features, features.t())
#         return G.div(a * b * c * d)

#     def forward(self, input):
#         G = self.gram_matrix(input)
#         self.loss = F.mse_loss(G, self.target)
#         return input