#!/usr/bin/env python
# coding=utf-8


# for image downloading
URL_FILE = "/Users/cheng/Data/data/nvzh_item_image_part_test"
URL_PREFIX = "https://img.alicdn.com/imgextra/"
IMAGE_PATH = "/Users/cheng/Data/data/nvzh_item_images"
CLASS_NAME = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41']
CLASS_NAME_MAP = {'T恤': '1', '学生校服': '12', '礼服/晚装': '27', '西装': '36', '牛仔裤': '23', '职业女裙套装': '29', '其它制服/套装': '6', '医护制服': '7', '毛衣': '20', '职业女裤套装': '30', '酒店工作制服': '39', '裙子': '34', '皮草': '24', '衬衫': '33', '风衣': '40', '婚纱': '11', '中老年女装': '3', '大码女装': '10', '马夹': '41', '毛呢外套': '19', '连衣裙': '38', '棉衣/棉服': '17', '皮衣': '25', '棉裤/羽绒裤': '18', '背心吊带': '31', '西装裤/正装裤': '37', '休闲裤': '4', '短外套': '26', '羽绒服': '28', '蕾丝衫/雪纺衫': '32', '旗袍': '15', '毛针织衫': '21', '民族服装/舞台装': '22', '上衣': '2', '卫衣/绒衫': '9', '抹胸': '14', '打底裤': '13', '裤子': '35', '半身裙': '8', '休闲运动套装': '5', '时尚套装': '16'}

# For image processing
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

# For dataset split
SPLIT_RATIO = {'train': 0.7, 'validation': 0.05, 'test': 0.25}
TRAIN_FILE = "/Users/cheng/Data/data/item_images/train_file"
VALIDATION_FILE = "/Users/cheng/Data/data/item_images/validate_file"
TEST_FILE = "/Users/cheng/Data/data/item_images/test_file"

# For logging


# For model
MODEL_DIR = ""
LOG_DIR = ""


# For training
BATCH_SIZE = 4

# For debug
DEBUG = True

class ModeConf(object):
    num_step = 100000
    leakiness = 0.01
    use_bottleneck = False
    num_classes = 4
    batch_size = 128
    weight_decay_rate = 0.002
    lrn_rate = 0.1
    lr_decay_steps = 20000
    optimizer = 'sgd'