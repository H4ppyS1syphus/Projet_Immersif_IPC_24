import random
import os
import torch
from torchvision import transforms as T
from PIL import Image, ImageDraw
import helper_tools.utils as utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from helper_tools.dataset import PlantDocDataset


# Define class names based on the PlantDoc dataset
class_names = [
    "background",  # Index 0 for the background
    "apple_scab", "apple_black_rot", "apple_healthy", 
    "bell_pepper_bacterial_spot", "bell_pepper_healthy",
    "cherry_healthy", "cherry_powdery_mildew",
    "corn_cercospora_leaf_spot_gray_leaf_spot", "corn_common_rust", "corn_healthy", "corn_northern_leaf_blight",
    "grape_black_rot", "grape_healthy", "grape_isariopsis_leaf_spot",
    "peach_bacterial_spot", "peach_healthy", 
    "potato_early_blight", "potato_healthy", "potato_late_blight", 
    "raspberry_healthy",
    "soybean_healthy",
    "squash_powdery_mildew",
    "strawberry_healthy", "strawberry_leaf_scorch",
    "tomato_bacterial_spot", "tomato_early_blight", "tomato_healthy", "tomato_late_blight", 
    "tomato_leaf_mold", "tomato_septoria_leaf_spot", "tomato_spider_mites", "tomato_target_spot", 
    "tomato_yellow_leaf_curl_virus", "tomato_mosaic_virus"
]

num_classes = len(class_names)  # Background + number of disease categories

def draw_predictions_on_image(image, predictions, class_names):
    """
    Draws bounding boxes and class labels on the image using predictions.
    """
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']

    # Convert image to PIL Image for drawing
    image = T.ToPILImage()(image).convert("RGB")
    draw = ImageDraw.Draw(image)

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        label = labels[i].item()
        score = scores[i].item()

        # Draw bounding box and label
        draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=2)
        draw.text((x_min, y_min), f"{class_names[label]}: {score:.2f}", fill="white")

    return image


def save_side_by_side(original_img, processed_img, save_path):
    """
    Combines original and processed images side by side and saves the result.
    """
    combined_img = Image.new("RGB", (original_img.width + processed_img.width, original_img.height))
    combined_img.paste(original_img, (0, 0))
    combined_img.paste(processed_img, (original_img.width, 0))
    combined_img.save(save_path)


def apply_nms_and_filter(predictions, iou_threshold=0.5, score_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) and filter out low-confidence boxes.
    """
    boxes = predictions["boxes"]
    scores = predictions["scores"]
    labels = predictions["labels"]
    print(labels)

    # Filter out low-confidence predictions
    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Apply Non-Maximum Suppression
    keep_nms = torch.ops.torchvision.nms(boxes, scores, iou_threshold)

    # Return only the filtered results
    return {"boxes": boxes[keep_nms], "scores": scores[keep_nms], "labels": labels[keep_nms]}


def main():
    model_path = "resnet_finetuned_plantdoc_new_20.pth"  # Path to the saved model

    # Load model and set to eval mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = utils.get_model_instance_segmentation(num_classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Load dataset
    dataset = PlantDocDataset(root='data/dataset-repo/TEST', 
                              csv_file='data/dataset-repo/test_labels.csv', 
                              transforms=utils.get_transform(train=False))

    # Select 10 random images
    indices = random.sample(range(len(dataset)), 10)
    
    # Process and visualize each image
    for i, idx in enumerate(indices):
        image, target = dataset[idx]

        # Move the image to the device and pass through the model
        with torch.no_grad():
            predictions = model([image.to(device)])[0]

        # Apply NMS and filter out low-confidence boxes
        predictions = apply_nms_and_filter(predictions, iou_threshold=0.1, score_threshold=0.5)

        # Convert image to PIL for drawing
        original_img = T.ToPILImage()(image).convert("RGB")
        processed_img = draw_predictions_on_image(image, predictions, class_names)

        # Save the original and processed side-by-side
        save_path = f"output_image_{i+1}.png"
        save_side_by_side(original_img, processed_img, save_path)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
