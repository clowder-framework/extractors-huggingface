import torch
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import ray


@ray.remote
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


if __name__ == "__main__":
    ray.init(_temp_dir="/taiga/mohanar2/segment-anything/ray")

    # Create a Ray actor
    segment_anything = SegmentAnything.options(num_gpus=1).remote()

    mask_json = ray.get(segment_anything.generate_mask.remote("test.jpeg"))
    ray.get(segment_anything.save_output.remote(mask_json, "test.jpeg", "output.png"))
    ray.shutdown()
