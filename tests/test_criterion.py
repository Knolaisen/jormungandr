import pytest
from transformers.image_transforms import center_to_corners_format
import torch
from jormungandr.training.criterion import CIoULoss, GIoULoss
from transformers.loss.loss_for_object_detection import (
    HungarianMatcher,
    ForObjectDetectionLoss as GIoULoss,
    generalized_box_iou,
)


@pytest.mark.parametrize(
    "pred_boxes",
    [
        torch.tensor([[0, 0, -1, -1]]),  # Bounding box completely outside the image
        torch.tensor(
            [[0.5, 0.5, -0.001, -0.001]]
        ),  # Bounding box completely outside the image
        torch.tensor([[-1, -1, 0.1, 0.1]]),  # Bounding box completely outside the image
        torch.tensor(
            [[0.5, 0.5, 0.2, 0.2]]
        ),  # Bounding box completely inside the image
        torch.tensor(
            [[1.5, 1.5, 0.2, 0.2]]
        ),  # Bounding box completely outside the image
    ],
)
def test_bounding_box_out_of_image(pred_boxes):

    # Test case 1: Bounding box completely outside the image
    # giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

    # pred_boxes = torch.tensor([[0, 0, -1, -1]])  # (x_center, y_center, width, height)
    target_boxes = torch.tensor(
        [[0.5, 0.5, 0.2, 0.2]]
    )  # (x_center, y_center, width, height)

    giou_cost = -generalized_box_iou(
        center_to_corners_format(pred_boxes), center_to_corners_format(target_boxes)
    )
