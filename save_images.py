import os
import matplotlib.pyplot as plt
import numpy as np

def save_image_from_array(image_array, filename, output_dir="generated_images"):
    """
    Saves a 2D numpy array as a PNG image.

    Args:
        image_array (np.ndarray): The 2D numpy array representing the image.
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

    plt.imshow(image_array, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0)
    plt.close()
