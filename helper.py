import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transform
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torchvision.models import efficientnet_b4
from ultralytics import YOLO

yolo_model = YOLO("weights/best.pt")
device = torch.device("cpu")

test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                 torchvision.transforms.ToTensor(),
                                                 transform.Normalize((0.5, 0.5, 0.5),
                                                                     (0.5, 0.5, 0.5))])

# test_model = torch.load('Effect/best.pt', map_location=device)
# test_model.eval()
model = efficientnet_b4(pretrained=False)
classifier = nn.Sequential(
    nn.Linear(
        in_features=model.classifier[1].in_features, out_features=256, bias=True),
    nn.Linear(in_features=256, out_features=27, bias=True)
)
model.classifier = classifier

# Load state dict
state_dict = torch.load('v1/best.pt', map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


def combine_2model(img_path):
    results = yolo_model([img_path])  # return a list of Results objects
    im = Image.open(img_path)
    im = np.array(im)  # read image as numpy array
    # Process results list
    for result in results:
        boxes = result.boxes.xyxy  # Boxes object for bbox outputs
        boxes = boxes.cpu().numpy()
        for box in boxes:

            box = np.reshape(box, (4))
            xmin, ymin, xmax, ymax = box
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            # crop cái bounding box từ ảnh ban đầu
            cropped = im[ymin:ymax, xmin:xmax]
            plt.subplot(2, 1, 1)
            plt.imshow(im)
            plt.subplot(2, 1, 2)
            plt.imshow(cropped)
            plt.show()
            cropped = Image.fromarray(cropped)  # convert to PIL
            # get the cropped to the classification model
            prediction = test_transform(cropped).to(
                device).view(1, 3, 224, 224)
            prediction = model(prediction)
            prediction = torch.argmax(prediction, dim=1)
            return prediction.cpu().item()


def combine_2model_v1(img_path):
    results = yolo_model([img_path])  # return a list of Results objects
    im = Image.open(img_path)
    im = np.array(im)  # read image as numpy array
    # Process results list
    for result in results:
        boxes = result.boxes.xyxy  # Boxes object for bbox outputs
        boxes = boxes.cpu().numpy()
        for box in boxes:

            box = np.reshape(box, (4))
            xmin, ymin, xmax, ymax = box
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            # crop cái bounding box từ ảnh ban đầu
            cropped = im[ymin:ymax, xmin:xmax]
            fig1, ax1 = plt.subplots()
            ax1.imshow(im)
            ax1.set_title('Original Image')

            # Second figure for the cropped image
            fig2, ax2 = plt.subplots()
            ax2.imshow(cropped)
            ax2.set_title('Cropped Image')

            cropped = Image.fromarray(cropped)  # convert to PIL
            # get the cropped to the classification model
            prediction = test_transform(cropped).to(
                device).view(1, 3, 224, 224)
            prediction = model(prediction)
            prediction = torch.argmax(prediction, dim=1)
            return fig1, fig2, prediction.cpu().item()


logger = logging.getLogger(__name__)


def combine_2model_v11(img_path):
    try:
        # Load and check image
        if not os.path.exists(img_path):
            logger.error(f"Image file not found: {img_path}")
            return None

        # Run YOLO detection
        logger.info("Running YOLO detection")
        results = yolo_model([img_path])

        # Load original image
        im = Image.open(img_path)
        im = np.array(im)

        # Check if any detections
        for result in results:
            boxes = result.boxes.xyxy
            if boxes.shape[0] == 0:  # No detections
                logger.warning("No logos detected in image")
                return None

            boxes = boxes.cpu().numpy()

            # Get the box with highest confidence (first box)
            box = boxes[0]  # Take first detection if multiple found

            try:
                # Convert coordinates to integers
                xmin, ymin, xmax, ymax = map(int, box[:4])

                # Ensure coordinates are within image bounds
                height, width = im.shape[:2]
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(width, xmax)
                ymax = min(height, ymax)

                # Crop the image
                cropped = im[ymin:ymax, xmin:xmax]

                if cropped.size == 0:  # Check if crop is empty
                    logger.error("Cropped region is empty")
                    return None

                # Create visualization
                fig1, ax1 = plt.subplots()
                ax1.imshow(im)
                ax1.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                            fill=False, color='red', linewidth=2))
                ax1.set_title('Detected Logo')
                ax1.axis('off')

                fig2, ax2 = plt.subplots()
                ax2.imshow(cropped)
                ax2.set_title('Cropped Logo')
                ax2.axis('off')

                # Convert to PIL for classification
                cropped_pil = Image.fromarray(cropped)

                # Run classification
                logger.info("Running classification model")
                with torch.no_grad():
                    prediction = test_transform(cropped_pil).to(
                        device).view(1, 3, 224, 224)
                    prediction = model(prediction)
                    predicted_class = torch.argmax(
                        prediction, dim=1).cpu().item()

                logger.info(
                    f"Classification complete. Predicted class: {predicted_class}")
                return fig1, fig2, predicted_class

            except Exception as e:
                logger.error(f"Error processing detection: {str(e)}")
                return None

        logger.warning("No valid detections found")
        return None

    except Exception as e:
        logger.error(f"Error in combine_2model_v1: {str(e)}", exc_info=True)
        return None
