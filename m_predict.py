import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from demo.predictor import VisualizationDemo


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    args.config_file = "configs/COCO-Detection.back/faster_rcnn_R_101_FPN_3x.yaml"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.INPUT.MAX_SIZE_TRAIN = 1280
    cfg.INPUT.MAX_SIZE_TEST = 1280
    cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 类别数
    # cfg.MODEL.WEIGHTS = "/home/Documents/pretrainedModel/Detectron2/R-50.pkl"  # 预训练模型权重
    cfg.MODEL.WEIGHTS = r'/home/dl/sgf/projects/detectron2_old/output_back2/model_final.pth'  # 最终权重
    cfg.SOLVER.IMS_PER_BATCH = 1  # batch_size=2; iteration = 1434/batch_size = 717 iters in one epoch
    ITERS_IN_ONE_EPOCH = int(1434 / cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 12) - 1  # 12 epochs
    cfg.SOLVER.BASE_LR = 0.002
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
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        # default="configs/quick_schedules/fast_rcnn_R_50_FPN_instant_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":

    has_target_imgs_folder = r'/home/dl/sgf/projects/detectron2_old/output_back2/shandong_test_result/has'
    no_target_imgs_folder = r'/home/dl/sgf/projects/detectron2_old/output_back2/shandong_test_result/no'

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    log_fullpath = r'/home/dl/sgf/projects/detectron2_old/output_back2/log'
    log_f = open(log_fullpath, 'w')
    demo = VisualizationDemo(cfg, instance_mode=ColorMode.SEGMENTATION)
    jpgs_folder =r'/home/dl/sgf/sd_test_img'
    # jpgs_folder =r'/home/dl/sgf/projects/detectron2_old/data/VOCdevkit_shandong_oilsite/COCO/train'
    output_folder = r'/home/dl/sgf/sd_test_result'
    jpgs_filenames = [filename for filename in os.listdir(jpgs_folder) if filename.endswith('.jpg')]
    os.makedirs(output_folder, exist_ok=True)
    # for path in tqdm.tqdm(args.input, disable=not args.output):
    for img_filename in jpgs_filenames:
        img_fullpath = os.path.join(jpgs_folder, img_filename)
        # use PIL, to be consistent with evaluation
        # img_fullName = os.path.join(INPUT_IMG_PATH, imgfile)
        img = read_image(img_fullpath, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        instances = predictions['instances']
        if len(instances) > 0:
            boxes = list(map(lambda x: x.cpu().detach().numpy().tolist(), instances._fields['pred_boxes'].tensor))
            scores = list(map(lambda x: x.cpu().item(), instances._fields['scores']))
            pred_classes = list(map(lambda x: x.cpu().item(), instances._fields['pred_classes']))
            pred_num = len(boxes)
            for i in range(pred_num):
                bbox = [int(c) for c in boxes[i]]
                record = f'{img_filename} {scores[i]} {pred_classes[i]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n'
                log_f.write(record)
                # log_f.close()
                # exit(0)
                print(record)

            # print(predictions)
            output_fullpath = os.path.join(has_target_imgs_folder, img_filename)
            visualized_output.save(output_fullpath)
        else:
            output_fullpath = os.path.join(no_target_imgs_folder, img_filename)
            visualized_output.save(output_fullpath)
        # print()
        # logger.info(
        #     "{}: detected {} instances in {:.2f}s".format(
        #         imgfile, len(predictions["instances"]), time.time() - start_time
        #     )
        # )

        # if args.output:
        #     if os.path.isdir(args.output):
        #         assert os.path.isdir(args.output), args.output
        #         out_filename = os.path.join(args.output, os.path.basename(imgfile))
        #     else:
        #         assert len(args.input) == 1, "Please specify a directory with args.output"
        #         out_filename = args.output
        #     visualized_output.save(out_filename)
        # elserere
        
        #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        #     cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
        #   utpu  if cv2.waitKey(0) == 27:
        #         break  # esc to quit


    log_f.close()