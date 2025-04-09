import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

# def get_dinov2(name="dinov2-base", **kwargs):
#     """
#     Loads a DINOv2 model from Huggingface and wraps it in a DinoEncoder.
    
#     Args:
#         model_name (str): The identifier for the DINOv2 model on Huggingface.
#         **kwargs: Any additional keyword arguments.
    
#     Returns:
#         nn.Module: A wrapped DINOv2 model ready to accept (B, C, H, W) image tensors.
#     """
#     from transformers import AutoImageProcessor, AutoModel
#     processor = AutoImageProcessor.from_pretrained(f"facebook/{name}")
#     model = AutoModel.from_pretrained(f"facebook/{name}")

#     class DinoEncoder(nn.Module):
#         def __init__(self, model, processor):
#             super().__init__()
#             self.model = model
#             self.processor = processor
#             self.to_pil = transforms.ToPILImage()

#         def forward(self, x):
#             """
#             Expects x of shape (B, C, H, W) in range [0, 1]. Converts each image to a PIL image,
#             applies the feature extractor, and then passes the results through the model.
#             Returns the feature vector for each image.
#             """
#             # Convert each image in the batch to a PIL image.
#             images = [self.to_pil(img) for img in x]
#             inputs = self.processor(images=images, return_tensors="pt")

#             inputs = {k: v for k, v in inputs.items()}
#             outputs = self.model(**inputs)

#             # Option A: Use CLS token embedding
#             features = outputs.pooler_output

#             # Option B: Average over all tokens
#             # features = outputs.last_hidden_state.mean(dim=1)

#             return features

#     return DinoEncoder(model, processor)

def get_dino(name="dinov2_vits14"):
    return torch.hub.load('facebookresearch/dinov2', name)


def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model
