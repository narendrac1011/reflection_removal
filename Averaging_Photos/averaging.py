import numpy as np
import cv2
import argparse
import os
import glob

# The Averaging function 
def averaging(files):

    # Initialize a variable `total_sum` with an array of zeros of the same size as the first image
    total_sum = np.zeros_like(cv2.imread(files[0], cv2.IMREAD_COLOR), dtype=np.float64)

    # Iterate over each file in the list of files
    for file in files:

        # Add the current image to the total sum
        total_sum += cv2.imread(file, cv2.IMREAD_COLOR).astype(np.float64)

    # Calculate the averaged image and converting it back into uint8 datatype
    averaged_image = (total_sum / len(files)).astype(np.uint8)
    return averaged_image


# Read the image file
def process_images(image_directory):
    image_files = glob.glob(os.path.join(image_directory, "*"))

    if not image_files:
        print("No images found in the directory.")
        return
    
    average_image = averaging(image_files)
    cv2.imshow("Averaged Image", average_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Averaged_Image_Set4.png', average_image)

# Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Averaging images to remove reflection")
    parser.add_argument("-i", "--imagepath", required = True, help = "Path to input image directory")
    return parser.parse_args()

# Main
if __name__ == "__main__":
    args = parse_arguments()
    image_directory = args.imagepath
    process_images(image_directory)