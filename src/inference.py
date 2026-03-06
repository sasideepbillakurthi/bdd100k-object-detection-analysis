from src.models.yolov8 import run_inference


run_inference(
    model_path="weights/best.pt",
    image_path="data/images/val/b1ca8418-84a133a0.jpg",
    output_dir="outputs/predictions"
)