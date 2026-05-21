import pytest
from transformers.image_transforms import center_to_corners_format
import torch
from transformers.loss.loss_for_object_detection import (
    generalized_box_iou,
)

from jormungandr.config.configuration import LossConfig
from jormungandr.training.criterion import build_criterion


@pytest.mark.parametrize(
    "pred_boxes",
    [
        # Bounding box completely outside the image
        # torch.tensor([[0, 0, -1, -1]]), # FAIL
        # Bounding box with zero area at the top-left corner
        torch.tensor([[0, 0, 0, 0]]),
        # Bounding box completely outside the image
        # torch.tensor([[0.5, 0.5, -0.001, -0.001]]), # FAIL
        torch.tensor([[0, 0, 1.1, 1.1]]),
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

    # Act
    generalized_box_iou(
        center_to_corners_format(pred_boxes), center_to_corners_format(target_boxes)
    )


def test_giou_loss_perfect_single_class_100_labels():
    """GIoU loss must collapse to ~0 when num_labels=100, every GT is the
    same single foreground class, and the model predicts every class and
    every box correctly (extra queries confidently say no-object).

    Sanity check for the MOT17-style scenario where the effective label
    space is one class out of a much larger output head.
    """
    num_labels = 100
    target_class = 1
    no_object_index = num_labels  # ImageLoss uses the last slot for "no object"

    batch_size = 2
    num_queries = 10
    num_targets = 3

    gt_boxes = torch.tensor(
        [
            [0.20, 0.20, 0.10, 0.10],
            [0.50, 0.50, 0.15, 0.15],
            [0.80, 0.30, 0.08, 0.20],
        ],
        dtype=torch.float32,
    )
    gt_class_labels = torch.full((num_targets,), target_class, dtype=torch.int64)
    labels = [
        {"class_labels": gt_class_labels.clone(), "boxes": gt_boxes.clone()}
        for _ in range(batch_size)
    ]

    # Confident logits: target_class for the first `num_targets` queries,
    # no-object for the remaining queries. Magnitude 50 makes softmax saturate
    # so CE drops below 1e-20.
    LARGE = 50.0
    logits = torch.full(
        (batch_size, num_queries, num_labels + 1), -LARGE, dtype=torch.float32
    )
    logits[:, :num_targets, target_class] = LARGE
    logits[:, num_targets:, no_object_index] = LARGE

    pred_boxes = torch.zeros((batch_size, num_queries, 4), dtype=torch.float32)
    pred_boxes[:, :num_targets] = gt_boxes
    # Unmatched queries get a deterministic far-away dummy box; they're
    # masked out of the bbox/GIoU loss because the matcher won't pick them.
    pred_boxes[:, num_targets:] = torch.tensor([0.05, 0.05, 0.02, 0.02])

    config = LossConfig(num_labels=num_labels, auxiliary_loss=False)
    giou_loss = build_criterion("giouloss")
    total_loss, loss_dict, aux = giou_loss(
        logits=logits,
        labels=labels,
        device=torch.device("cpu"),
        pred_boxes=pred_boxes,
        config=config,
    )

    assert aux is None
    assert loss_dict["loss_giou"].item() == pytest.approx(0.0, abs=1e-6)
    assert loss_dict["loss_bbox"].item() == pytest.approx(0.0, abs=1e-6)
    assert loss_dict["loss_ce"].item() == pytest.approx(0.0, abs=1e-6)
    assert loss_dict["cardinality_error"].item() == pytest.approx(0.0, abs=1e-6)
    assert total_loss.item() == pytest.approx(0.0, abs=1e-5)
