import random
from pathlib import Path

import cv2
import torch
import numpy as np

from src.parser import load_annotations
from src.dataset import BDDDetectionDataset
from src.config import IMAGE_DIR_VAL, LABEL_FILE_VAL, DETECTION_CLASSES
from src.models.swin_faster_rcnn import build_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw_boxes(image, boxes, labels, color, prefix):

    for box, label in zip(boxes, labels):

        x1, y1, x2, y2 = map(int, box)

        class_name = DETECTION_CLASSES[label - 1]

        cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)

        cv2.putText(
            image,
            f"{prefix}:{class_name}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    return image


def qualitative_evaluation(num_samples=20):

    annotations = load_annotations(LABEL_FILE_VAL)

    dataset = BDDDetectionDataset(IMAGE_DIR_VAL, annotations)

    model = build_model()

    model.to(DEVICE)
    model.eval()

    output_dir = Path("outputs/qualitative")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = random.sample(dataset.image_ids, num_samples)

    with torch.no_grad():

        for image_id in sample_ids:

            image = dataset.load_image(image_id)
            anns = dataset.get_annotations(image_id)

            img_tensor = torch.from_numpy(image).permute(2,0,1) / 255.0
            img_tensor = img_tensor.float().to(DEVICE)

            output = model([img_tensor])[0]

            pred_boxes = output["boxes"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()

            keep = scores > 0.5
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]

            gt_boxes = []
            gt_labels = []

            for ann in anns:
                b = ann.bbox
                gt_boxes.append([b.x1, b.y1, b.x2, b.y2])
                gt_labels.append(DETECTION_CLASSES.index(ann.category) + 1)

            gt_boxes = np.array(gt_boxes)
            gt_labels = np.array(gt_labels)

            vis = image.copy()

            vis = draw_boxes(vis, gt_boxes, gt_labels, (0,255,0), "GT")
            vis = draw_boxes(vis, pred_boxes, pred_labels, (0,0,255), "PRED")

            save_path = output_dir / f"{image_id}.jpg"

            cv2.imwrite(str(save_path), vis)

            print(f"Saved {save_path}")


if __name__ == "__main__":

    qualitative_evaluation(num_samples=10)