from data_io import *
from data_augmentation import *
from metrics import *
from model import *

from sklearn.model_selection import train_test_split

datamanager = DataManager()

X_train, Y_train, coverage = datamanager.load_train()

ious_by_split = []
thresholds_by_split = []

seeds = [24, 42, 1997, 63]
for i in range(1):
    print('-'*5,'Start Model #', i, '-'*5)
    x_train, x_valid, y_train, y_valid, cov_train, cov_valid = train_test_split(
        X_train, Y_train, coverage, test_size=0.2, stratify=coverage[:, 1], random_state=seeds[i])

    x_train_, y_train_ = augment_images(x_train, y_train)

    y_train_ = np.piecewise(y_train_, [y_train_ > 125, y_train_ < 125], [1, 0])

    y_valid = np.piecewise(y_valid, [y_valid > 125, y_valid < 125], [1, 0])


    amodel = create_model(datamanager.im_height, datamanager.im_width, datamanager.im_chan)
    history = fit(amodel, x_train_, y_train_, x_valid, y_valid, 'model_testing_script_{}'.format(i))

    """
    model = load_model('model_testing_script_{}.h5'.format(i))


    preds_valid = model.predict(x_valid).reshape(-1,
                                                 datamanager.im_height, datamanager.im_width)

    thresholds = np.linspace(0, 1, 50)
    ious = np.array([iou_metric_batch(y_valid, np.int32(preds_valid > threshold)) for threshold in tqdm(thresholds)])

    threshold_best_index = np.argmax(ious[9:-10]) + 9
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    print("Threshold ", threshold_best, 'Ious', iou_best)
    """
