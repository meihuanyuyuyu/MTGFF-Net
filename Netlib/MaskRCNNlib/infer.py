import argparse
import Config.mrcnn_configs
import torch

parser = argparse.ArgumentParser(description="Mask RCNN Infering")
parser.add_argument('--config', type=str, default='config/ade20k-resnet50dilated-ppm_deepsup.yaml', help='config file')
parser.add_argument('--model_path', type=str, default='exp/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth', help='path to the trained model')
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--ds", type=str, default="conic")
parser.add_argument("--train_idx", type=int, default=1)
process_arg = parser.parse_args()

arg = getattr(Config.mrcnn_configs, process_arg.config)(process_arg.ds, process_arg.device, Config.mrcnn_configs.AnchorWH)

class InferEngine:
    def __init__(self) -> None:
        print("Initliazing InferEngine...")
        print("Check setted Config:\n")
        for _ in dir(arg):
            if not _.startswith("_"):
                print(_, "=", getattr(arg, _))
        
        net = arg.model(
            anchors=arg.anchor_wh.to(arg.device),
            backbone=arg.BACKBONE,
            bottom_up=arg.BOTTOM_UP,
            proposal_generator=arg.PROPAOSAL_GENERATOR,
            stride=arg.STRIDE,
            rpn_pos_threshold=arg.RPN_POS_THRESHOLD,
            rpn_fraction_ratio=arg.RPN_FRACTION_RATIO,
            nms_threshold=arg.NMS_THRESHOLD,
            pre_nms_k=arg.PRE_NMS_K,
            post_nms_k=arg.POST_NMS_K,
            roi_head=arg.ROI_HEAD,
            box_detection=arg.BOX_DETECTION,
            expand=arg.EXPAND,
            expand_ratio=arg.EXPAND_RATIO,
            use_gt_box=arg.USE_GT_BOX,
            roi_resolution=arg.ROI_RESOLUTION,
            stage2_max_proposal=arg.STAGE2_MAX_PROPOSAL,
            stage2_sample_ratio=arg.STAGE2_SAMPLE_RATIO,
            box_weight=arg.BOX_WEIGHT,
            roi_pos_threshold=arg.ROI_POS_THRESHOLD,
            post_decttion_score_threshold=arg.POST_DETECTION_SCORE_THRESHOLD,
            detection_per_img=arg.DETECTION_PER_IMG,
            num_classes=arg.NUM_CLASSES,
            use_semantic=arg.USE_SEMANTIC,
            seg_stride=arg.SEG_STRIDE,
            fuse_feature=arg.FUSE_FEATURE,
        ).to(device=arg.device)
        net.load_state_dict(torch.load(process_arg.model_path))
        net.eval()
        net.to(arg.device)
    
    def infer(self, img):
        pass

    def load_data(self):
        splitted_idx = [1,2,3]
        if arg.ds == 'conic':
            from tools.dataset import MRCNNLizardDataset
            test_sets = [_ for _ in splitted_idx if _ != arg.train_idx]
            test_set1 = MRCNNLizardDataset(data_path='data/Lizard',)
            test_set2 = MRCNNLizardDataset()

