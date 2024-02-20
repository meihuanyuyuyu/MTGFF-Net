from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances,load_coco_json

register_coco_instances('conic_train_1',{}, json_file='data/CoNIC_Challenge/train_1/annotations/instance_annotations.json', image_root='data/CoNIC_Challenge/train_1/images')
register_coco_instances('conic_train_2',{}, json_file='data/CoNIC_Challenge/train_2/annotations/instance_annotations.json', image_root='data/CoNIC_Challenge/train_2/images')
register_coco_instances('conic_train_3',{}, json_file='data/CoNIC_Challenge/train_3/annotations/instance_annotations.json', image_root='data/CoNIC_Challenge/train_3/images')
register_coco_instances('conic_val_1',{}, json_file='data/CoNIC_Challenge/val_1/annotations/instance_annotations.json', image_root='data/CoNIC_Challenge/val_1/images')
register_coco_instances('conic_val_2',{}, json_file='data/CoNIC_Challenge/val_2/annotations/instance_annotations.json', image_root='data/CoNIC_Challenge/val_2/images')
register_coco_instances('conic_val_3',{}, json_file='data/CoNIC_Challenge/val_3/annotations/instance_annotations.json', image_root='data/CoNIC_Challenge/val_3/images')
register_coco_instances('conic_test_1',{}, json_file='data/CoNIC_Challenge/test_1/annotations/instance_annotations.json', image_root='data/CoNIC_Challenge/test_1/images')
register_coco_instances('conic_test_2',{}, json_file='data/CoNIC_Challenge/test_2/annotations/instance_annotations.json', image_root='data/CoNIC_Challenge/test_2/images')
register_coco_instances('conic_test_3',{}, json_file='data/CoNIC_Challenge/test_3/annotations/instance_annotations.json', image_root='data/CoNIC_Challenge/test_3/images')


def conic_dataset_function(ds_name,json_fp, image_root):
    out_list=load_coco_json(json_fp, image_root, dataset_name=ds_name, extra_annotation_keys=None)
    return out_list

print(DatasetCatalog.get('conic_train_1')[0])