# source - https://www.freecodecamp.org/news/use-segment-anything-model-to-create-masks/
import torch
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint="sam_vit_h_4b8939.pth").to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

# Give the path of your image
IMAGE_PATH = 'test.jpeg'
# Read the image from the path
image = cv2.imread(IMAGE_PATH)
# Convert to RGB format (cv2 uses BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate segmentation mask
sam_result = mask_generator.generate(image_rgb)

def show_output(result_dict, original_image):
    output_image = original_image.copy()  # Make a copy of the original to overlay masks
    sorted_result = sorted(result_dict, key=(lambda x: x['area']), reverse=True)
    # Overlay each segment with a random color
    for val in sorted_result:
        mask = val['segmentation']
        color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)  # Random color
        output_image[mask == 1] = output_image[mask == 1] * 0.5 + color_mask * 0.5
    return output_image

# Create the output image
output_image = show_output(sam_result, image_rgb)
output_image_bgr = cv2.cvtColor(output_image.astype('uint8'), cv2.COLOR_RGB2BGR)  # Convert to BGR for saving with OpenCV
cv2.imwrite('output.png', output_image_bgr)
