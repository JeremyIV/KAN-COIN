import argparse
import datetime
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from KAN import *
from quantization import *

parser = argparse.ArgumentParser()

parser.add_argument(
    "-ni",
    "--num_iters",
    help="Number of iterations to train for",
    type=int,
    default=1000,
)
parser.add_argument(
    "-lr", "--learning_rate", help="Learning rate", type=float, default=0.001
)
parser.add_argument(
    "-ds",
    "--dataset_dir",
    help="directory with the training images",
    type=str,
    default="kodak-dataset",
)
parser.add_argument(
    "-iid",
    "--image_id",
    help="Image ID to train on, if not the full dataset",
    type=int,
    default=3,
)
parser.add_argument(
    "-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=15
)
parser.add_argument(
    "-nl", "--num_layers", help="Number of layers", type=int, default=3
)
parser.add_argument(
    "-gs",
    "--grid_size",
    help="Spline grid size.",
    type=int,
    default=32,
)

parser.add_argument(
    "--logdir",
    type=str,
    default="logs/",
    help="directory in which to save the training results."
)

args = parser.parse_args()

###################################
## Load the image
###################################

img_filename = str(args.image_id).zfill(2) + ".png"
img_path = Path(args.dataset_dir) / img_filename
img_pil = Image.open(img_path)

width, height = img_pil.size

if width > height:
    xmin, xmax = -1,1
    ratio = height / width
    ymin, ymax = -ratio, ratio
else:
    ymin, ymax = -1,1
    ratio = width / height
    xmin, xmax = -ratio, ratio


features = torch.tensor(np.array(img_pil)).reshape(-1, 3).to('cuda') / 255
rows, cols = torch.meshgrid(torch.linspace(ymin, ymax, height), torch.linspace(xmin, xmax, width))
coordinates = torch.stack([rows, cols], axis=-1).reshape(-1,2).to('cuda')

dataset = {
    'train_input': coordinates,
    'train_label': features
}

###################################
## Create the network
###################################

layers = [2] + [args.layer_size] * args.num_layers + [3]
model = KAN(layers, grid_size=args.grid_size).to('cuda')

###################################
## Train the network
###################################

train_log = model.train(dataset, steps=args.num_iters, lr=args.learning_rate, batch=10_000, device='cuda')

###################################
## Save the results
###################################
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path(args.logdir) / f"results_{timestamp}"
results_dir.mkdir(parents=True, exist_ok=True)

# Save the training log
train_log_path = results_dir / "train_log.pkl"
with train_log_path.open("wb") as f:
    pickle.dump(train_log, f)

# Save the model weights
model_path = results_dir / "model.pth"
torch.save(model.state_dict(), model_path)

# Save the reconstructions at full precision and quantized precision
def get_psnr(img1, img2):
    mse = np.mean((np.array(img1).astype('float') - np.array(img2).astype('float'))**2)
    return 10 * np.log10(255**2 / mse)


reconstruction = model.predict_in_batches(dataset['train_input'])
recon_img = Image.fromarray((reconstruction.reshape(height, width, 3).clip(0,1).detach().cpu().numpy() * 255).astype('uint8'))
recon_img.save(results_dir / "fp_reconstruction.png")

print(f"unqantized PSNR: {get_psnr(recon_img, img_pil)}")

q_kan = quantize_and_dequantize(model, bits=9)

reconstruction = q_kan.predict_in_batches(dataset['train_input'])
recon_img = Image.fromarray((reconstruction.reshape(height, width, 3).clip(0,1).detach().cpu().numpy() * 255).astype('uint8'))
recon_img.save(results_dir / "quantized_reconstruction.png")

print(f"quantized PSNR: {get_psnr(recon_img, img_pil)}")
num_params = q_kan.count_trainable_params()
num_bits = num_params * 9
num_pixels = width * height
bpp = num_bits / num_pixels
print(f"quantized BPP: {bpp:.02f}")