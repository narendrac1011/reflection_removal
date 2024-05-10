import numpy as np
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import argparse
import torch
from tqdm import tqdm
from model import GCNet

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
padsize = 150

# Class to Test the Image Dataset
class TestImageDataset(Dataset):
    def __init__(self, root):
        self.tensor_setup = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.files = sorted(glob.glob(os.path.join(root, "*.*")))

    def __getitem__(self, index):
        file_path = self.files[index % len(self.files)]
        image = np.array(Image.open(file_path), 'f') / 255.
        image = np.pad(image, [(padsize, padsize), (padsize, padsize), (0, 0)], 'symmetric')
        return {"image": self.tensor_setup(image[:, :, :3]), "name": os.path.basename(file_path).split(".")[0]}

    def __len__(self):
        return len(self.files)

# Convert to Numpy function
def convert_to_numpy(tensor, H, W):
    image = tensor[:, :, padsize:H - padsize, padsize:W - padsize].clone()
    input_numpy = image[:, :, :H, :W].clone().cpu().numpy().reshape(3, H - padsize * 2, W - padsize * 2).transpose(1, 2, 0)
    for i in range(3):
        input_numpy[:, :, i] = input_numpy[:, :, i] * std[i] + mean[i]
    return input_numpy

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--path", required=True, help="Path to input dataset")
opt = parser.parse_args()
dataset_name = opt.path
os.makedirs(os.path.join(dataset_name, "output"), exist_ok=True)

# Initializing the GCNet Model
print("Creating Model")
generator = GCNet()

# Loading the Pre-trained Weights
print("Loading weights")
generator.load_state_dict(torch.load("GCNet_weight.pth", map_location="cpu"))

# Loading the Testing Dataset
print("Loading dataset")
image_dataset = TestImageDataset(dataset_name)
print("[Dataset: %s] --> %d images" % (dataset_name, len(image_dataset)))

# Go through each image in the testing dataset
for image_num in tqdm(range(len(image_dataset))):

    # Preparing the image for inference by padding it and reshaping it
    data = image_dataset[image_num]
    image = data["image"]
    _, first_h, first_w = image.size()
    image = torch.nn.functional.pad(image, (0, (image.size(2) // 16) * 16 + 16 - image.size(2), 0, (image.size(1) // 16) * 16 + 16 - image.size(1)), "constant")
    image = image.view(1, 3, image.size(1), image.size(2))

    # Removing reflection from the image using the Generator
    print("Removing Reflection from Image")
    with torch.no_grad():
        output = generator(image)
    output_np = np.clip(convert_to_numpy(output, first_h, first_w) + 0.015, 0, 1)
    input_np = convert_to_numpy(image, first_h, first_w)
    final_output = np.fmin(output_np, input_np)

    # Saving the processed image to the output directory
    print("Saving Image")
    Image.fromarray(np.uint8(final_output * 255)).save(os.path.join(dataset_name, "output", data["name"] + ".png"))
