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
        # Bounding box completely outside the image
        torch.tensor([[0, 0, -1, -1]]),
        # Bounding box with zero area at the top-left corner
        torch.tensor([[0, 0, 0, 0]]),
        # Bounding box completely outside the image
        torch.tensor([[0.5, 0.5, -0.001, -0.001]]),
        # Bounding box completely outside the image
        torch.tensor([[-1, -1, 0.1, 0.1]]),
        # Bounding box completely inside the image
        torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        # Bounding box completely outside the image
        torch.tensor([[1.5, 1.5, 0.2, 0.2]]),
    ],
)
def test_bounding_box_out_of_image(pred_boxes):
    # pred_boxes (x_center, y_center, width, height)
    target_boxes = torch.tensor(
        [[0.5, 0.5, 0.2, 0.2]]
    )  # (x_center, y_center, width, height)

    giou_cost = -generalized_box_iou(
        center_to_corners_format(pred_boxes), center_to_corners_format(target_boxes)
    )
