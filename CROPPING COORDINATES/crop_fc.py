from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import math,os

def draw_square(image_path, center_x, center_y, side_length):
    """
    Draw a square on the image centered at (center_x, center_y) with the given side length.

    :param image_path: Path to the input image
    :param center_x: X-coordinate of the center of the square
    :param center_y: Y-coordinate of the center of the square
    :param side_length: Side length of the square
    :return: Image with the square drawn on it
    """
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Calculate the coordinates of the top-left and bottom-right corners of the square
    left = center_x - side_length // 2
    top = center_y - side_length // 2
    right = center_x + side_length // 2
    bottom = center_y + side_length // 2

    # Draw the square
    draw.rectangle([left, top, right, bottom], outline="red", width=2)

    return image
    
def crop_square(image_path, output_path, center_x, center_y, side_length):
    """
    Crop a square portion of the image centered at (center_x, center_y) with the given side length.

    :param image_path: Path to the input image
    :param output_path: Path to save the cropped image
    :param center_x: X-coordinate of the center of the square
    :param center_y: Y-coordinate of the center of the square
    :param side_length: Side length of the square
    """
    # Open the image
    image = Image.open(image_path)

    # Calculate the coordinates of the top-left corner of the square
    left = center_x - side_length // 2
    top = center_y - side_length // 2
    right = center_x + side_length // 2
    bottom = center_y + side_length // 2

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))

    # Save the cropped image
    cropped_image.save(output_path)


# Example usage
image_path = os.path.join(non_fc_dir,'5.png')
image = Image.open(image_path)
draw_square(image_path, 1150, 670, 400) # 1200, 650, 500


cwd = os.getcwd()
input_dir = fc_dir
images = os.listdir(input_dir)

center_x = 1150 # X-coordinate of the center of the square
center_y =  670  # Y-coordinate of the center of the square
side_length = 400  # Side length of the square

##----------------------------------------------SAME_WILL_BE_DONE_FOR_NON_FC_IMAGES---------------------------------------##
output_dir = os.path.join(cwd,'Processed-FC-Dataset','FC')

for i,image in enumerate(images):
    fid = image.split(".")[0]
    # if fid == '21941' or fid == '112':
    input_path = os.path.join(input_dir,image)
    output_path = os.path.join(output_dir,image)
    crop_square(input_path,output_path,center_x,center_y,side_length)
    print(f'{image} Done')
    # if i%50 == 0:
        # print(f'\r{i+1} Done',end='')