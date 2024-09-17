import os
import torch
import argparse
from torchvision import transforms as T
from PIL import Image, ImageDraw
import plant_detection.helper_tools.utils as utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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
    "raspberry_healthy", "soybean_healthy", "squash_powdery_mildew",
    "strawberry_healthy", "strawberry_leaf_scorch", "tomato_bacterial_spot", 
    "tomato_early_blight", "tomato_healthy", "tomato_late_blight", 
    "tomato_leaf_mold", "tomato_septoria_leaf_spot", "tomato_spider_mites", 
    "tomato_target_spot", "tomato_yellow_leaf_curl_virus", "tomato_mosaic_virus"
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


def apply_nms_and_filter(predictions, iou_threshold=0.5, score_threshold=0.4):
    """
    Apply Non-Maximum Suppression (NMS) and filter out low-confidence boxes.
    Also combine boxes into a single box if needed.
    """
    boxes = predictions["boxes"]
    scores = predictions["scores"]
    labels = predictions["labels"]

    # Filter out low-confidence predictions
    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Check if there are any remaining boxes
    if len(boxes) == 0:
        return {"boxes": torch.tensor([]), "scores": torch.tensor([]), "labels": torch.tensor([])}

    # Apply Non-Maximum Suppression if there are multiple boxes
    if len(boxes) > 1:
        keep_nms = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
        boxes = boxes[keep_nms]
        scores = scores[keep_nms]
        labels = labels[keep_nms]

    # If there's still more than one box, combine them into a single box that covers all the boxes
    if len(boxes) > 1:
        x_min = torch.min(boxes[:, 0])
        y_min = torch.min(boxes[:, 1])
        x_max = torch.max(boxes[:, 2])
        y_max = torch.max(boxes[:, 3])

        # Create a single bounding box that covers all the boxes
        boxes = torch.tensor([[x_min, y_min, x_max, y_max]], device=boxes.device)
        scores = torch.tensor([torch.max(scores)], device=scores.device)
        labels = torch.tensor([labels[0]], device=labels.device)  # Assuming all boxes have the same label

    return {"boxes": boxes, "scores": scores, "labels": labels}



def detect_plant(image_path):
    model_path = "./plant_detection/resnet_finetuned_plantdoc_new_20.pth"  # Path to the saved model

    # Load model and set to eval mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = utils.get_model_instance_segmentation(num_classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make predictions
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # Apply NMS and filter out low-confidence boxes
    predictions = apply_nms_and_filter(predictions, iou_threshold=0.5, score_threshold=0.4)

    # Draw bounding boxes and labels on the image
    processed_img = draw_predictions_on_image(image_tensor.squeeze(0), predictions, class_names)

    # Save the processed image
    save_path = f"output_{os.path.basename(image_path)}"
    processed_img.save(save_path)
    print(f"Processed image saved as: {save_path}")
    return (processed_img, predictions, class_names) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect plants in an image using a trained model.")
    parser.add_argument('--image', type=str, required=True, help="Path to the image file.")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} does not exist.")
        exit(1)

    detect_plant(args.image)
