import os
import cv2
import logging
from collections import OrderedDict

import torch

import detectron2.utils.comm as comm
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA

import warnings

warnings.filterwarnings('ignore')

# 数据集路径
# DATASET_ROOT = 'data/voc2coco'
DATASET_ROOT = 'data/COCO-handi'
DATASET_ROOT = 'data/VOCdevkit_shandong_oilsite/COCO'

# DATASET_ROOT = '/home/dl/Documents/detectron2_old/data/new_TankDataset/VOC2007'
# ANN_ROOT = os.path.join(DATASET_ROOT, 'coco2017')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'train')
# TRAIN_PATH = os.path.join(DATASET_ROOT, 'train')
# VAL_PATH = os.path.join(DATASET_ROOT, 'val')
TEST_PATH = os.path.join(DATASET_ROOT, 'test')

TRAIN_JSON = os.path.join(DATASET_ROOT, 'annotations/train.json')
TEST_JSON = os.path.join(DATASET_ROOT, 'annotations/test.json')
# VAL_JSON = os.path.join(DATASET_ROOT, 'annotations/voc2coco_val.json')


# 数据集类别元数据
# DATASET_CATEGORIES = [{'name': 'dog', 'id': 18, 'isthing': 1, 'color': [220, 20, 60]}, {'name': 'hot dog', 'id': 58, 'isthing': 1, 'color': [20, 220, 60]}]
DATASET_CATEGORIES = [
    {'name': 'handi', 'id': 1, 'isthing': 1, 'color': [220, 20, 60]}
]
DATASET_CATEGORIES = [
    {'name': 'oilsite', 'id': 1, 'isthing': 1, 'color': [220, 20, 60]}
]
# 数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "train_2021": (TRAIN_PATH, TRAIN_JSON),
    "test_2021": (TEST_PATH,TEST_JSON),
    # "val_2019": (VAL_PATH, VAL_JSON),
}


def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   metadate=get_dataset_instances_meta(),
                                   json_file=json_file,
                                   image_root=image_root)


def get_dataset_instances_meta():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_dataset_instances(name, metadate, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco",
                                  **metadate)


# 注册数据集和元数据
def plain_register_dataset():
    DatasetCatalog.register("train_2021", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH, "train_2021"))
    MetadataCatalog.get("train_2021").set(thing_classes=['oilsite'],
                                                    json_file=TRAIN_JSON,
                                                    image_root=TRAIN_PATH)
    # DatasetCatalog.register("val_2019", lambda: load_coco_json(VAL_JSON, VAL_PATH, "val_2019"))
    # MetadataCatalog.get("val_2019").set(thing_classes=['oilwell'],
    #                                             json_file=VAL_JSON,
    #                                             image_root=VAL_PATH)
    DatasetCatalog.register("test_2021", lambda: load_coco_json(TEST_JSON, TEST_PATH, "test_2021"))
    MetadataCatalog.get("test_2021").set(thing_classes=['oilsite'],
                                        json_file=TEST_JSON,
                                        image_root=TEST_PATH)


# 查看数据集标注
def checkout_dataset_annotation(name="test_2021"):
    dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow('show', vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()  # 拷贝default config副本
    args.config_file = "configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    # args.eval_only = True
    # args.training = False
    # args.config_file = "configs/COCO-Detection.back/faster_rcnn_R_50_DC5_3x.yaml"

    cfg.merge_from_file(args.config_file)   # 从config file 覆盖配置
    # cfg.merge_from_list(args.opts)          # 从CLI参数 覆盖配置

    # 更改配置参数
    cfg.DATASETS.TRAIN = ("train_2021",)
    cfg.DATASETS.TEST = ("test_2021",)
    cfg.DATALOADER.NUM_WORKERS = 0  # 单线程
    cfg.INPUT.MAX_SIZE_TRAIN = 1280
    cfg.INPUT.MAX_SIZE_TEST = 1280
    cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # 类别数
    cfg.MODEL.WEIGHTS = "models/Faster R-CNN R 101 FPN.pkl"    # 预训练模型权重
    cfg.MODEL.WEIGHTS = "/home/dl/sgf/projects/detectron2_old/output/model_0011192.pth"    # 预训练模型权重
    # cfg.MODEL.WEIGHTS = r"/home/dl/sgf/projects/detectron2_old/output/model_final.pth"    # 预训练模型权重
    cfg.SOLVER.IMS_PER_BATCH = 2  # batch_size=2; iters_in_one_epoch = dataset_imgs/batch_size
    ITERS_IN_ONE_EPOCH = int( 1725/ cfg.SOLVER.IMS_PER_BATCH)
    print(ITERS_IN_ONE_EPOCH)
    cfg.SOLVER.MAX_ITER = 35000  # (ITERS_IN_ONE_EPOCH * 122 ) - 1 # 12 epochs
    cfg.SOLVER.MAX_ITER = 20000  # (ITERS_IN_ONE_EPOCH * 122 ) - 1 # 12 epochs
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (30000,)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    print(cfg)

    # 注册数据集
    register_dataset()

    # if args.eval_only:
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        input_image = torch.rand((1, 3, 100, 100))
        input_images = [input_image]
        # model().training = False
        o = model(input_images)
        # 导出onnx格式
        print()

        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        num_gpus_per_machine=1,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

