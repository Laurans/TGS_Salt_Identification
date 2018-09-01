from itertools import chain
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

def best_iou_and_threshold(y_true, y_pred, plot=False, shortcut=False):
    if shortcut:
        thresholds = [0.5]
        ious = np.array([iou_metric_batch(y_true, np.int32(y_pred > threshold)) for threshold in thresholds])
    else:
        thresholds = np.linspace(0, 1, 50)
        ious = np.array([iou_metric_batch(y_true, np.int32(y_pred > threshold)) for threshold in tqdm(thresholds, desc='iou and threshold')])

    if shortcut:
        threshold_best_index = 0
    else:
        threshold_best_index = np.argmax(ious[9:-10]) + 9
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    if plot:
        plt.switch_backend('agg')
        plt.plot(thresholds, ious)
        plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
        plt.xlabel("Threshold")
        plt.ylabel("IoU")
        plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
        plt.legend()
        plt.savefig(fname='iou_threshold.png')
        print('threshold_best: {}'.format(threshold_best))
        print('iou_best: {}'.format(iou_best))

    return iou_best, threshold_best

def plot_prediction(imgs, masks, preds, fname='sanity_check_prediction.png', image_id=[]):
    max_images = 48
    grid_width = 12
    grid_height = int(max_images / grid_width)*3
    plt.switch_backend('agg')
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width*2, grid_height*2))
    imgs = np.squeeze(imgs[:max_images])
    masks = np.squeeze(masks[:max_images])
    preds = np.squeeze(preds[:max_images])
    for i, (img, mask, pred) in enumerate(zip(imgs, masks, preds)):
        ax_image = axs[int(i / grid_width)*3, i % grid_width]
        ax_image.imshow(img, cmap="Greys")
        if image_id != []:
            ax_image.set_title('{}'.format(image_id[i]))
        else:
            ax_image.set_title("Image")
        ax_image.set_yticklabels([])
        ax_image.set_xticklabels([])
        ax_mask = axs[int(i / grid_width)*3+1, i % grid_width]
        ax_mask.imshow(img, cmap="Greys")
        ax_mask.imshow(mask, alpha=0.9, cmap="Greens")
        ax_mask.set_title("Mask")
        ax_mask.set_yticklabels([])
        ax_mask.set_xticklabels([])
        ax_pred = axs[int(i / grid_width)*3+2, i % grid_width]
        ax_pred.imshow(img, cmap="Greys")
        ax_pred.imshow(pred, alpha=0.9, cmap="Blues")
        coverage_pred = np.sum(pred) / pow(101, 2)
        ax_pred.set_title("Predict")
        ax_pred.set_yticklabels([])
        ax_pred.set_xticklabels([])

    plt.savefig(fname=fname)
    plt.close(fig)

def plot_hist(y_true, y_pred):
    values = [iou_metric(y_, y_hat) for y_, y_hat in zip(tqdm(y_true, desc='iou for hist'), y_pred)]
    plt.hist(values)
    plt.ylabel('Count images')
    plt.xlabel('IOU')
    plt.savefig(fname='hist_iou_value.png')
    plt.close()
    return values
