import sys
sys.path.append('models/research/')
from detection_model import *
import os
from tensorflow import keras
from keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from numpy import newaxis
import glob
import json
from collections import namedtuple
from pathlib import Path
import matplotlib.pyplot as plt

in_files_train = []
in_files_train_ann = []

in_files_test = []
in_files_test_ann = []

validate_training = False

if (len(sys.argv) == 3):
    print('Only training set of images and annotations are provided')
    print('Proceeding with only these set of data')
    image_data_dir = sys.argv[1]
    in_files_train = glob.glob(image_data_dir+'/*.jpg')
    in_files_train_ann = sys.argv[2]
elif (len(sys.argv) == 5): 
    image_data_dir = sys.argv[1]
    in_files_train = glob.glob(image_data_dir+'/*.jpg')
    in_files_train_ann = sys.argv[2]
    image_data_dir = sys.argv[3]
    in_files_test = glob.glob(image_data_dir+'/*.jpg')
    in_files_test_ann = sys.argv[4]
    validate_training = True
else:
    print('Program tries to rescore the prediction scores predicted by alredy existing object detectors')
    print('Usuage:')
    print('main_prog.py <train images folder> <traing images annotations> <val images folder> <val images annotations >')
    print('                     or')
    print('main_prog.py <train images folder> <traing images annotations>')
    print('Coco data set is used for this exercise.')
    print('Data set and annotations can be downloaded from https://cocodataset.org/#download')
    print('ssd_mobilenet_v1_coco_2017_11_17 model is used for object detection purposes')
    print('Tensor flow models can be downloaded using git clone --depth 1 https://github.com/tensorflow/models')
    exit()

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

coco_cats = np.arange(1, 91).reshape(-1,1)
def ohe(num):
    idx = np.where(coco_cats == num)
    return [1 if i == idx[0] else 0 for i in range(80)]

input_features = []
input_features_tst = []
pred_json = []
file_names = []

for j in in_files_train:
    nm = Path(j).stem + '.jpg'
    file_names.append(nm)
    img = np.array(Image.open(j))
    W, H = img.shape[1], img.shape[0]
    if (len(img.shape) < 3):
        img = np.stack((img,)*3, axis = -1)
    predictions = run_inference_for_single_image(detection_model, img)
    for i in range(len(predictions['detection_classes'])):
        x, y, w, h = predictions['detection_boxes'][i]
        bb_params = np.array([x/W, y/H, w/W, h/W], dtype=float)
        scores = np.array(float(predictions['detection_scores'][i]), dtype=float)
        clss = np.array(ohe(int(predictions['detection_classes'][i])), dtype=float)
        in_f = np.concatenate((scores,clss, bb_params), axis = None)
        input_features_tst.append(in_f)
        stat = {"category_id":int(predictions['detection_classes'][i]),
	        "bbox":predictions['detection_boxes'][i].tolist(),
	        "score":float(predictions['detection_scores'][i]),
	    }
        if (stat["score"] > 0.5):
            pred_json.append(stat)
            input_features.append(in_f)

print(len(input_features), len(input_features_tst))
#print(file_names)
input_features = np.array(input_features)
#input_features = pad_sequences(input_features, dtype=float)
in_features = tf.constant(input_features)

#Load the annotations for the test images
f_train = open(in_files_train_ann, 'r')
data = json.load(f_train)
categories = data['categories']
super_cats = {cat["id"]: cat["supercategory"] for cat in categories}
categories = {cat["id"]: cat for cat in categories}
category_index = list(categories.keys())

imgs = data["images"]
images = dict()
img_id = []
for img in imgs:
    # for nm in file_names:
    #     if (nm == img["file_name"]):
    #         img_id.append(img["id"])
    #         file_names.remove(nm)
    img.pop("license", None)
    img.pop("date_captured", None)
    img.pop("flickr_url", None)
    images[img["id"]] = img

annotations = []
ann_dict = {}
j = 0
if "annotations" in data.keys():
    anns = data["annotations"]
    for ann in anns:
        ann.pop("segmentation", None)
        ann.pop("area", None)
        i = ann["image_id"]
        annotations.append(ann)
        ann_dict["i"] = j
        j = j+1


detects = len(pred_json)
len_ann = len(annotations)
target_features_res = []
#target_features = []
cvrd = np.zeros(len_ann, dtype=bool)

def IoU(a, b):
    x1, y1, w1, h1 = a["bbox"]
    x2, y2, w2, h2 = b["bbox"]
    # overlap = cal_overlap(x1, y1, w1, h1, x2, y2, w2, h2)
    # un = w1 * h1 + w2 * h2 - overlap
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(w1, w2)
    yB = min(h1, h2)

    iA = max(0, xA - xB + 1) * max(0, yA - yB + 1)
    bxAA = (w1 - x1 + 1) * (h1 - y1 + 1)
    bxBA = (w2 - x2 + 1) * (h2 - y2 + 1)
    iou = iA / float(bxAA + bxBA - iA)
    return iou

def IoU_matrix(a,b):
    len_a, len_b = len(a), len(b)
    iou = np.zeros((len_b, len_a))
    for lb in range(len_a):
        for la in range(len_a):
            iou[lb, la] = IoU(a[la], b[lb])
    return iou

threshold = 0.5
cnt = 0
for i, pred in enumerate(pred_json):
    gts_class = []
    track_mat = []
    for j in range(len_ann):
        # if (annotations[j]["category_id"] == pred["category_id"]):
        #     th = IoU(pred, annotations[j])
        #     if (th > 0.5):
        #         print(th)
        #         print( pred["score"])
        if(annotations[j]["category_id"] == pred["category_id"] and 
                not cvrd[j] and IoU(pred, annotations[j]) >= threshold):
                #and annotations[j]["iscrowd"]):
            gts_class.append(annotations[j])
            track_mat.append(j)

    if len(gts_class) == 0:
        cnt = cnt + 1
        target_features_res.append(0)
        continue

    class_ious = IoU_matrix([pred], gts_class)
    matched_gt_class = class_ious.argmax()
    matched_gt = track_mat[matched_gt_class]
    cvrd[matched_gt] = True
    #target_features.append(pred["score"])
    target_features_res.append(pred["score"])
    # print('adding scores to target')

print(cnt)
#target_features = np.array(target_features)
# if len(target_features) > 0:
#     print(len(target_features))
#     target_features = pad_sequences(target_features, dtype=float)
#     print(len(target_features))
#target_f = tf.constant(target_features)
target_features = tf.constant(target_features_res)
tf.reshape(target_features, [-1, 1])

#Model
early_stopping = EarlyStopping(min_delta = 0.001,
                                         patience = 20,
                                         restore_best_weights=True)

model = keras.Sequential()
emb = model.add(layers.Embedding(input_dim=85, output_dim=85))

model.add(layers.SimpleRNN(85, input_shape=in_features.shape[1:]))
model.add(layers.Reshape((1,85)))
model.add(layers.Bidirectional(layers.GRU(85, return_sequences=True)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))
model.add(layers.SimpleRNN(85))		
model.add(layers.Dropout(0.3))
model.add(layers.BatchNormalization())
model.add(layers.Dense(80, activation='relu')) 

print(model.summary())

model.compile(optimizer='Adam', loss="mse", metrics=["accuracy"])
hist = model.fit(in_features, target_features, epochs=30, batch_size=128)
hist_dict = hist.history
hist_dict.keys()

print(hist_dict.keys())
loss = hist.history['loss']
acc = hist.history['accuracy']
epochs = range(1, len(loss) + 1)

print("Training Results")
print("Training Loss: ", loss[-1])
print("Accuracy: ", acc[-1])

res = []
pred = model.predict(in_features)
fp = 0
tp = 0
for i in range(len(pred)):
    res.append(np.amax(pred[i]))
    if (res[i] > target_features_res[i]):
        tp = tp + 1
    elif (res[i] < target_features_res[i]):
        fp = fp + 1

print("True Positives :", tp)
print("False positives:", fp)

#Validation
if(validate_training == True):
    input_features_val = []
    pred_json_val = []

    for j in in_files_test:
        #print(j)
        img = np.array(Image.open(j))
        W, H = img.shape[1], img.shape[0]
        if (len(img.shape) < 3):
            img = np.stack((img,)*3, axis = -1)
        predictions = run_inference_for_single_image(detection_model, img)
        for i in range(len(predictions['detection_classes'])):
            x, y, w, h = predictions['detection_boxes'][i]
            bb_params = np.array([x/W, y/H, w/W, h/W], dtype=float)
            scores = np.array(float(predictions['detection_scores'][i]), dtype=float)
            clss = np.array(ohe(int(predictions['detection_classes'][i])), dtype=float)
            in_f = np.concatenate((scores,clss, bb_params), axis = None)
            stat = {"category_id":int(predictions['detection_classes'][i]),
                "bbox":predictions['detection_boxes'][i].tolist(),
                "score":float(predictions['detection_scores'][i]),
            }
            if (stat["score"] > 0.5):
                pred_json_val.append(stat)
                input_features_val.append(in_f)

    input_features_val = np.array(input_features_val)
    in_features_val = tf.constant(input_features_val)

    #Load the annotations for the test images

    f_val = open(in_files_test_ann, 'r')
    data = json.load(f_val)
    categories = data['categories']
    super_cats = {cat["id"]: cat["supercategory"] for cat in categories}
    categories = {cat["id"]: cat for cat in categories}
    category_index = list(categories.keys())

    imgs = data["images"]
    images = dict()
    for img in imgs:
        img.pop("license", None)
        img.pop("date_captured", None)
        img.pop("flickr_url", None)
        images[img["id"]] = img

    annotations_val = []
    ann_dict_val = {}
    j = 0
    if "annotations" in data.keys():
        anns = data["annotations"]
        for ann in anns:
            ann.pop("segmentation", None)
            ann.pop("area", None)
            i = ann["image_id"]
            annotations_val.append(ann)
            ann_dict_val["i"] = j
            j = j+1


    detects = len(pred_json_val)
    len_ann = len(annotations_val)
    target_features_val = tf.zeros(detects)
    cvrd_val = np.zeros(len_ann, dtype=bool)

    threshold = 0.5

    for i, pred in enumerate(pred_json_val):
        gts_class = []
        track_mat = []
        for j in range(len_ann):
            if(annotations_val[j]["category_id"] == pred["category_id"] and 
                    not cvrd_val[j] and IoU(pred, annotations_val[j]) >= threshold
                    and annotations_val[j]["iscrowd"]):
                gts_class.append(annotations_val[j])
                track_mat.append(j)

        if len(gts_class) == 0:
            continue

        class_ious = IoU_matrix([pred], gts_class)
        matched_gt_class = class_ious.argmax()
        matched_gt = track_mat[matched_gt_class]
        cvrd_val[matched_gt] = True
        target_features_val[i] = pred["score"]

    tf.reshape(target_features_val, [-1, 1])
    hist_val = model.evaluate(in_features_val, target_features_val)
    loss = hist.history['loss']
    acc = hist.history['accuracy']

