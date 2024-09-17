#main 
from src.extraction import get_image, convert_rgb2hsv, create_mask, print_raw_and_extracted
from plant_detection.detect_plant import detect_plant
from disease_detection.eval import eval
import numpy as np
from PIL import Image

def main():
    # 1. Load the image
    image_path = "./plant_detection/image.png"

    processed_img, predictions, class_names = detect_plant(image_path)
    processed_img.save("output_after_first_AI.jpg")

    # Extract the bounding box coordinates and convert them to integers
    box_tensor = predictions['boxes'][0]
    box = box_tensor.detach().cpu().numpy().astype(int)

    # Crop the image using the bounding box
    image = processed_img.crop((box[0], box[1], box[2], box[3]))
    image.save("output_after_crop.jpg")

    # 2. Convert the image to HSV color space
    image_hsv = convert_rgb2hsv(image)

    lower_green = np.array([0.08, 0.08, 0.08])
    upper_green = np.array([0.5, 1.0, 1.0])

    # 3. Create a mask for green regions
    mask = create_mask(image_hsv, lower_green, upper_green)

    # 4. Extract the green regions
    extracted_array = np.array(image)
    extracted_array[~mask] = 0  # Set non-green pixels to black
    extracted = Image.fromarray(extracted_array.astype('uint8'))
    extracted.save("output_after_extraction.jpg")

    # 7. Evaluate the plant health
    health_status = eval(extracted)
    print(f"Plant Health Status: {health_status}")

if __name__ == "__main__":
    main()