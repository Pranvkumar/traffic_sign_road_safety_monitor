import numpy as np
import pytest

from traffic_monitor.evaluation.metrics import (average_precision, box_iou, classification_metrics, confusion_matrix,
                                                detection_at_threshold, mask_metrics, match_detections)


def test_confusion_and_classification_by_hand():
    y, p = np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 1, 0])
    cm = confusion_matrix(y, p, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    m = classification_metrics(y, p, 3)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["per_class"][1]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"][0]["recall"] == pytest.approx(1 / 2)
    assert m["per_class"][2]["f1"] == 0.0


def test_box_iou_by_hand():
    iou = box_iou([[0, 0, 10, 10]], [[5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou, [[50 / 150, 0.0]])


def test_matching_is_one_to_one():
    gt = np.array([[0, 0, 10, 10]])
    preds = np.array([[0, 0, 10, 10], [1, 1, 10, 10]])
    tp = match_detections(preds, np.array([0.9, 0.8]), gt, 0.5)
    assert tp.tolist() == [True, False]


def test_missed_sign_rate_and_fp_per_image():
    per_image = [(np.array([[0, 0, 10, 10]]), np.array([0.9]), np.array([[0, 0, 10, 10], [50, 50, 60, 60]])),
                 (np.array([[0, 0, 5, 5]]), np.array([0.8]), np.zeros((0, 4)))]
    r = detection_at_threshold(per_image, 0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert r["missed_sign_rate"] == pytest.approx(0.5)
    assert r["false_positives_per_image"] == pytest.approx(0.5)


def test_average_precision_perfect_and_half():
    gt = np.array([[0, 0, 10, 10]])
    assert average_precision([(gt, np.array([0.9]), gt)])["ap"] == pytest.approx(1.0)
    per = [(gt, np.array([0.9]), gt), (np.zeros((0, 4)), np.zeros(0), gt)]
    assert average_precision(per)["ap"] == pytest.approx(0.5)


def test_mask_iou_dice_with_ignore():
    yt = np.array([1, 1, 0, 0, 255])
    yp = np.array([1, 0, 0, 0, 1])
    m = mask_metrics(yt, yp, 2, ignore_index=255)
    assert m["per_class_iou"][1] == pytest.approx(1 / 2)
    assert m["per_class_dice"][1] == pytest.approx(2 / 3)
    assert m["pixel_accuracy"] == pytest.approx(3 / 4)
