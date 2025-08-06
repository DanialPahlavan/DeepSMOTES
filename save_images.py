import os
import matplotlib.pyplot as plt
import numpy as np

def save_image_from_array(image_array, filename, output_dir="generated_images"):
    """
    Saves a 2D or 3D numpy array as a PNG image.

    Args:
        image_array (np.ndarray): The numpy array representing the image.
        filename (str): The name of the output image file.
        output_dir (str): The directory to save the image in.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Reshape if necessary (e.g., from a flat array)
    if len(image_array.shape) == 1:
        # Assuming a square image, e.g., 28x28 for MNIST
        side = int(np.sqrt(len(image_array)))
        if side * side == len(image_array):
            image_array = image_array.reshape(side, side)
        else:
            print(f"Warning: Cannot reshape array of size {len(image_array)} into a square image.")
            return

    # If image is in (C, H, W) format, transpose to (H, W, C) for matplotlib
    if len(image_array.shape) == 3 and image_array.shape[0] in [1, 3]:
        image_array = image_array.transpose((1, 2, 0))

    # Squeeze single-channel images to 2D for cmap
    if len(image_array.shape) == 3 and image_array.shape[2] == 1:
        image_array = image_array.squeeze(axis=2)

    # Display the image
    plt.imshow(image_array, cmap='gray' if len(image_array.shape) == 2 else None)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0)
    plt.close()
