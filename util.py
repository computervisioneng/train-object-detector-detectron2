import os

from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2


def get_dicts(img_dir, ann_dir):
    dataset_dicts = []
    for idx, file in enumerate(os.listdir(ann_dir)):
        # annotations should be provided in yolo format

        record = {}

        filename = os.path.join(img_dir, file[:-4] + '.jpg')
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        with open(os.path.join(ann_dir, file)) as r:
            lines = [l[:-1] for l in r.readlines()]

        for _, line in enumerate(lines):
            if len(line) > 2:
                label, cx, cy, w_, h_ = line.split(' ')

                obj = {
                    "bbox": [int((float(cx) - (float(w_) / 2)) * width),
                             int((float(cy) - (float(h_) / 2)) * height),
                             int(float(w_) * width),
                             int(float(h_) * height)],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": int(label),
                }

                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_datasets(root_dir, class_list_file):

    with open(class_list_file, 'r') as reader:
        classes_ = [l[:-1] for l in reader.readlines()]

    for d in ['train']:

        DatasetCatalog.register(d, lambda d=d: get_dicts(os.path.join(root_dir, d, 'imgs'),
                                                         os.path.join(root_dir, d, 'anns')))
        MetadataCatalog.get(d).set(thing_classes=classes_)

    return len(classes_)


def train(output_dir,
          data_dir,
          class_list_file,
          learning_rate=0.00025,
          batch_size=4,
          max_iter=10000,
          checkpoint_period=500,
          device='cpu',
          model="COCO-Detection/retinanet_R_101_FPN_3x.yaml"):

    nmr_classes = register_datasets(data_dir, class_list_file)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = ("train",)

    if device in ['cpu']:
        cfg.MODEL.DEVICE = 'cpu'

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nmr_classes

    cfg.OUTPUT_DIR = output_dir

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
