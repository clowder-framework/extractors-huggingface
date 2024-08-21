import torch
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np

class SegmentAnything:
    def __init__(self):
        print("Initializing SegmentAnything")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_type = "vit_h"
        self.sam = sam_model_registry[self.model_type](
            checkpoint="/taiga/mohanar2/segment-anything/sam_vit_h_4b8939.pth").to(device=self.device)
        print("Gpu Available " + str(torch.cuda.is_available()))
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.predictor = SamPredictor(self.sam)
        print("SegmentAnything initialized")

    def generate_mask(self, image_path):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam_result = self.mask_generator.generate(image_rgb)
        return sam_result

    def save_output(self, result_dict, original_image_path, output_path):
        original_image = cv2.imread(original_image_path)
        output_image = original_image.copy()
        sorted_result = sorted(result_dict, key=(lambda x: x['area']), reverse=True)
        for val in sorted_result:
            mask = val['segmentation']
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            output_image[mask == 1] = output_image[mask == 1] * 0.5 + color_mask * 0.5

        output_image_bgr = cv2.cvtColor(output_image.astype('uint8'), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_image_bgr)

    def generate_prompt_mask(self, image_path, bounding_box):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        input_box = np.array(bounding_box)
        masks, _, _ = self.predictor.predict(box=input_box[None, :])
        return masks

    def save_prompt_output(self, masks, original_image_path, output_path):
        original_image = cv2.imread(original_image_path)
        output_image = original_image.copy()
        for mask in masks:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            output_image[mask == 1] = output_image[mask == 1] * 0.5 + color_mask * 0.5

        output_image_bgr = cv2.cvtColor(output_image.astype('uint8'), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_image_bgr)

if __name__ == "__main__":
    segment_anything = SegmentAnything()

    bounding_box = [157.26, 183.61, 266, 149]
    if bounding_box:
        mask = segment_anything.generate_prompt_mask("test.jpeg", bounding_box)
        segment_anything.save_prompt_output(mask, "test.jpeg", "output.png")
    else:
        mask_json = segment_anything.generate_mask("test.jpeg")
        segment_anything.save_output(mask_json, "test.jpeg", "output.png")
