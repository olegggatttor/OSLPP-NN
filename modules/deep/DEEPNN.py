from torchvision import models
import torch.nn as nn


def get_feature_extractor_model():
    resnet = models.resnet50(pretrained=True, progress=True).train()
    layers = list(resnet.children())[:-1]
    return layers


def fix_batch_normalization_layers(module):
    if isinstance(module, nn.modules.BatchNorm1d):
        module.eval()
    if isinstance(module, nn.modules.BatchNorm2d):
        module.eval()
    if isinstance(module, nn.modules.BatchNorm3d):
        module.eval()


def get_simple_deep_nn(n_classes):
    resnet = get_feature_extractor_model()
    model = nn.Sequential(*resnet, nn.Flatten(),

                          nn.Linear(2048, 1024, bias=True),
                          nn.LeakyReLU(inplace=True),

                          nn.Linear(1024, 512, bias=True),
                          nn.LeakyReLU(inplace=True),

                          nn.Linear(512, n_classes, bias=True))
    model.apply(fix_batch_normalization_layers)
    return model
