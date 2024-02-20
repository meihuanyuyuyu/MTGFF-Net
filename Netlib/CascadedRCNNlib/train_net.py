from .config.cascadedrcnn_config import _cfg
import logging
import os
import torch
from collections import OrderedDict
from detectron2.solver import build_optimizer,build_lr_scheduler
from torch.cuda.amp import autocast,GradScaler

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer,PeriodicCheckpointer
from detectron2.data import MetadataCatalog,build_detection_test_loader,build_detection_train_loader
from detectron2.engine import default_argument_parser, default_setup, launch, default_writers
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import GeneralizedRCNNWithTTA,build_model
from detectron2.utils.events import EventStorage

logger = logging.getLogger("detectron2")

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg,dataset_name)
        evaluator = build_evaluator(cfg,dataset_name,output_folder=None)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model:torch.nn.Module, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(model,cfg.OUTPUT_DIR,optimizer,scheduler)
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    ) # get the iteration number from the last checkpoint
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(checkpointer=checkpointer,period=cfg.SOLVER.CHECKPOINT_PERIOD,max_iter=max_iter)
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    scaler = GradScaler()
    with EventStorage(start_iter) as storage:
        for data,iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            optimizer.zero_grad()
            with autocast():
                loss_dict = model(data)
                losses = sum(loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            storage.put_scalars(total_loss=losses,**loss_dict)
            scheduler.step()

            if cfg.TEST.EVAL_PERIOD > 0 and iteration % cfg.TEST.EVAL_PERIOD == 0 and iteration != max_iter - 1:
                do_test(cfg, model)
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter - 1):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
                

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = _cfg.clone()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    do_train(cfg, model,resume=args.resume)
    return do_test(cfg, model)

    
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
