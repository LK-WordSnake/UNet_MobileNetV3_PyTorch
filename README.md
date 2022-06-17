mAP.py
import os
import sys
import numpy as np
import argparse
import time

import cv2
import torch
import torch.nn as nn
from torch.cuda import amp
from tqdm import tqdm

sys.path.append('../input/util-code/')
from make_model import make_model
from local_feat import make_local_model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

def create_model(backbone):
    model_backbone = {'R50': 'resnet50',
                      'RXt101ibn': 'resnext101_ibn_a',
                      'SER101ibn': 'se_resnet101_ibn_a',
                      'ResNeSt101': 'resnest101',
                      'ResNeSt269': 'resnest269'}

    # model = make_model(model_backbone[backbone])
    model = make_local_model(model_backbone[backbone])
    model_name = opt.weights
    model_path = os.path.join(opt.model_path, model_name)
    model.load_param(model_path)
    model = nn.DataParallel(model)
    model.to('cuda')
    model.eval()
    return model


def load_image(img_path):
    mean = torch.tensor([123.675, 116.280, 103.530]).to('cuda')
    std = torch.tensor([57.0, 57.0, 57.0]).to('cuda')

    img = cv2.imread(img_path)
    img = cv2.resize(img, (opt.img_size, opt.img_size))
    img = torch.tensor(img)
    img = img[:, :, [2, 1, 0]]
    img = torch.unsqueeze(img, dim=0).to('cuda')
    img = (img - mean) / std
    img = img.permute(0, 3, 1, 2)
    img = img.float()
    return img


def main():
    # 读取特征
    # info = torch.load(os.path.join('outputs', '{}_features_{}.pt'.format(opt.backbone, opt.data_code)), map_location='cuda')
    info = torch.load(os.path.join('outputs', '{}_features_{}_local.pt'.format(opt.backbone, opt.data_code)), map_location='cuda')
    features = info['features']
    # {image_path: feature, image_path: feature, image_path: feature ......}

    # 提取特征
    model = create_model(opt.backbone)

    mAP_info = []
    top_k = opt.top_k
    all_time = []
    for i in tqdm(os.listdir(opt.root)):
        code = i.split('_')[0]
        img_path = os.path.join(opt.root, i)
        start_time = time.time()
        img = load_image(img_path)

        with amp.autocast():
            feat = model(img)

        feat = feat / torch.norm(feat, 2, 1, keepdim=True)
        feat = feat.cpu().detach().numpy()
        end_time = time.time()
        result_sort = {}

        for p, f in features.items():
            score = np.dot(feat, f.T)[0] * 100
            result_sort[p] = score

        result_sort = sorted(result_sort.items(), key=lambda x: x[1], reverse=True)

        mAP_info_single = []
        count_all = 0
        count_query = 0

        # start_time = time.time()
        for i, (k, v) in enumerate(dict(result_sort[: top_k]).items()):
            count_all += 1
            res_img_code = k.split('/')[-1].split('_')[0]

            if res_img_code == code:
                count_query += 1
                mAP_info_single.append(count_query / count_all)

        # end_time = time.time()
        times = end_time - start_time
        all_time.append(times)

        try:
            mAP_info.append(sum(mAP_info_single) / count_query)
        except:
            mAP_info.append(sum(mAP_info_single) / 1)

    mAP = sum(mAP_info) / len(os.listdir(opt.root))
    timer = sum(all_time) / len(os.listdir(opt.root))
    print('mAP@{}: {}'.format(top_k, mAP))
    print(f'Average single img search time: {timer:.8f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 超参数
    parser.add_argument('--weights', type=str, default='resnest101_100_local.pth',
                        help='resnext101_ibn_a_100.pth'
                             'resnet50_100.pth'
                             'se_resnet101_ibn_a_100.pth'
                             'resnest101_100.pth'
                             'resnest101_2178.pth'
                             'resnest101_100_local.pth'
                             'se_resnet101_ibn_a_2178.pth'
                             'se_resnet101_ibn_a_local.pth'
                             'resnext101_ibn_a_local.pth')

    parser.add_argument('--backbone', type=str, default='ResNeSt101',
                        help='R50'                  # 256
                             'RXt101ibn'            # 384
                             'SER101ibn'            # 384
                             'ResNeSt101'           # 384
                             'ResNeSt269')          # 448

    parser.add_argument('--model_path', type=str, default='../input/models/')

    parser.add_argument('--img_size', type=int, default=384,
                        help='256, 384, 448, 512')

    parser.add_argument('--data_code', type=str, default='84')
    parser.add_argument('--top_k', type=int, default=29)
    parser.add_argument('--root', type=str, default='inference_84')
    opt = parser.parse_args()

    main()


draw_feature.py
import os
import sys
import argparse
import cv2
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from collections import OrderedDict

sys.path.append('../input/util-code/')
from make_model import make_model
# from local_feat import make_local_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "4"


class ImageDataset(Dataset):
    """Image Dataset."""

    def __init__(self, dataset, transforms):
        _ = transforms
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        supported = [".jpg", ".JPG", ".png", ".PNG"]
        if os.path.splitext(img_path)[-1] in supported:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (opt.img_size, opt.img_size))
            img = torch.tensor(img)
            img = img[:, :, [2, 1, 0]]
            return img, img_path


def val_collate_fn(batch):
    """Val collate fn."""

    imgs, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), img_paths


# 推理
class ReID_Inference:
    """ReID Inference."""

    def __init__(self, backbone):

        model_backbone = {'R50': 'resnet50',
                          'RXt101ibn': 'resnext101_ibn_a',
                          'SER101ibn': 'se_resnet101_ibn_a',
                          'ResNeSt101': 'resnest101',
                          'ResNeSt269': 'resnest269'}

        self.model = make_model(model_backbone[backbone])
        # self.model = make_local_model(model_backbone[backbone])
        model_name = opt.weights
        model_path = os.path.join(opt.model_path, model_name)
        self.model.load_param(model_path)
        self.model = nn.DataParallel(self.model)
        self.batch_size = opt.batch_size
        self.model.to('cuda')
        self.model.eval()
        self.mean = torch.tensor([123.675, 116.280, 103.530]).to('cuda')
        self.std = torch.tensor([57.0, 57.0, 57.0]).to('cuda')

    def extract(self, imgpath_list):
        """Extract feature for one image."""

        val_set = ImageDataset(imgpath_list, None)

        pin_memory = True
        num_workers = 8

        val_loader = DataLoader(
            val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=val_collate_fn,
            pin_memory=pin_memory
        )

        batch_res_dic = OrderedDict()
        for (batch_data, batch_path) in tqdm(val_loader,
                                             total=len(val_loader)):
            with torch.no_grad():
                batch_data = batch_data.to('cuda')
                batch_data = (batch_data - self.mean) / self.std
                batch_data = batch_data.permute(0, 3, 1, 2)
                batch_data = batch_data.float()
                with amp.autocast():
                    feat = self.model(batch_data)

                feat = feat / torch.norm(feat, 2, 1, keepdim=True)
                feat = feat.cpu().detach().numpy()
                # print(feat)

            for index, imgpath in enumerate(batch_path):
                batch_res_dic[imgpath] = feat[index]
        del val_loader, val_set, feat, batch_data
        return batch_res_dic


def main():
    # 提取特征
    reid = ReID_Inference(opt.backbone)

    info = []
    if opt.mode == 'query':
        # 获取query 特征
        for img in os.listdir(opt.query_root):
            img_path = os.path.join(opt.query_root, img)
            info.append(img_path)

        result = reid.extract(info)
        os.makedirs('outputs', exist_ok=True)
        torch.save({'features': result},
                   os.path.join('outputs', '{}_features_{}_query.pt'.format(opt.backbone, str(opt.data_code))))

    else:
        inference_info = os.listdir(opt.root)
        for cls in inference_info:
            cls_path = os.path.join(opt.root, cls)
            for img in os.listdir(cls_path):
                img_path = os.path.join(opt.root, cls, img)
                info.append(img_path)

        result = reid.extract(info)
        os.makedirs('outputs', exist_ok=True)
        torch.save({'features': result},
                   os.path.join('outputs', '{}_features_{}.pt'.format(opt.backbone, str(opt.data_code))))
        # torch.save({'features': result},
        #            os.path.join('outputs', '{}_features_{}_local.pt'.format(opt.backbone, str(opt.data_code))))

    # 读取特征
    # info = torch.load(os.path.join('outputs', 'features_park_519.pt'), map_location='cuda')
    # features = info['features']
    # print(features)
    # {image_path: feature, image_path: feature, image_path: feature ......}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 超参数
    # 带有1417标志代表在1417数据集上训练获得
    parser.add_argument('--weights', type=str, default='resnext101_ibn_a_2178.pth',
                        help='resnext101_ibn_a_100.pth'
                             'resnet50_100.pth'
                             'se_resnet101_ibn_a_100.pth'
                             'resnest101_100.pth'
                             'resnest101_2178.pth'
                             'resnest101_local_2178.pth'
                             'resnest101__2_local.pth'
                             'se_resnet101_ibn_a_2178.pth'
                             'se_resnet101_ibn_a_local.pth'
                             'resnext101_ibn_a_local.pth'
                             'resnext101_ibn_a_2178.pth')

    parser.add_argument('--backbone', type=str, default='RXt101ibn',
                        help='R50'  # 256
                             'RXt101ibn'  # 384
                             'SER101ibn'  # 384
                             'ResNeSt101'  # 384
                             'ResNeSt269')  # 448

    parser.add_argument('--img_size', type=int, default=384,
                        help='256, 384, 448, 512')

    parser.add_argument('--model_path', type=str, default='../input/models/')
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--data_code', type=str, default='84')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--query_root', type=str, default='inference_84')
    parser.add_argument('--root', type=str, default='../input/inference_84')
    # parser.add_argument('--root', type=str, default='../input/inference_484')
    # parser.add_argument('--root', type=str, default='../../../dataset_1417')
    opt = parser.parse_args()

    main()

val.py
import os
import sys
import numpy as np
import argparse

import cv2
import torch
import torch.nn as nn
from torch.cuda import amp
from PIL import Image
from tqdm import tqdm

sys.path.append('../input/util-code/')
from make_model import make_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def create_model(backbone):
    model_backbone = {'R50': 'resnet50',
                      'RXt101ibn': 'resnext101_ibn_a',
                      'SER101ibn': 'se_resnet101_ibn_a',
                      'ResNeSt101': 'resnest101',
                      'ResNeSt269': 'resnest269'}
    model = make_model(model_backbone[backbone])
    model_name = opt.weights
    model_path = os.path.join(opt.model_path, model_name)
    model.load_param(model_path)
    model = nn.DataParallel(model)
    model.to('cuda')
    model.eval()
    return model


def load_image(img_path):
    mean = torch.tensor([123.675, 116.280, 103.530]).to('cuda')
    std = torch.tensor([57.0, 57.0, 57.0]).to('cuda')

    img = cv2.imread(img_path)
    img = cv2.resize(img, (opt.img_size, opt.img_size))
    img = torch.tensor(img)
    img = img[:, :, [2, 1, 0]]
    img = torch.unsqueeze(img, dim=0).to('cuda')
    img = (img - mean) / std
    img = img.permute(0, 3, 1, 2)
    img = img.float()
    return img


def main():
    # 读取特征
    info = torch.load(os.path.join('outputs', '{}_features_{}.pt'.format(opt.backbone, opt.data_code)), map_location='cuda')
    features = info['features']
    # {image_path: feature, image_path: feature, image_path: feature ......}

    # 提取特征
    model = create_model(opt.backbone)

    os.makedirs(os.path.join(opt.out_dir, opt.backbone), exist_ok=True)
    for i in tqdm(os.listdir(opt.query_root)):
        code = i.split('.')[0]
        img_path = os.path.join(opt.query_root, code + '.jpg')
        img = load_image(img_path)

        with amp.autocast():
            feat = model(img)

        feat = feat / torch.norm(feat, 2, 1, keepdim=True)
        feat = feat.cpu().detach().numpy()
        count = 0
        result_sort = {}
        os.makedirs(os.path.join(opt.out_dir, opt.backbone, code), exist_ok=True)

        for p, f in features.items():
            score = np.dot(feat, f.T)[0] * 100
            result_sort[p] = score

        result_sort = sorted(result_sort.items(), key=lambda x: x[1], reverse=True)

        top_k = opt.top_k
        for i, (k, v) in enumerate(dict(result_sort[: top_k]).items()):

            count += 1
            org_img = Image.open(img_path)
            res_img = Image.open(k)
            res_img_name = k.split('/')[-1].split('.')[0]

            org_img.save(os.path.join(opt.out_dir, opt.backbone, code, 'query.jpg'))
            # # res_img.save(os.path.join('result', code, res_img_name + '_top{}.jpg'.format(str(i + 1))))
            res_img.save(os.path.join(opt.out_dir, opt.backbone, code, 'top{}.jpg'.format(str(i + 1))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 超参数
    parser.add_argument('--weights', type=str, default='resnest101_100.pth',
                        help='resnext101_ibn_a_100.pth'
                             'resnet50_100.pth'
                             'se_resnet101_ibn_a_100.pth'
                             'resnest101_100.pth'
                             '')

    parser.add_argument('--backbone', type=str, default='ResNeSt101',
                        help='R50'                  # 256
                             'RXt101ibn'            # 384
                             'SER101ibn'            # 384
                             'ResNeSt101'           # 384
                             'ResNeSt269')          # 448

    parser.add_argument('--img_size', type=int, default=384,
                        help='256, 384, 448, 512')

    parser.add_argument('--model_path', type=str, default='../input/models/')

    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--data_code', type=str, default='278')
    parser.add_argument('--query_root', type=str, default='inference_278')
    # parser.add_argument('--root', type=str, default='../input/inference_278')
    parser.add_argument('--out_dir', type=str, default='result')
    opt = parser.parse_args()

    main()

get_multi_feature.py
import os
import sys
import argparse
import cv2
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from collections import OrderedDict

sys.path.append('../input/util-code/')
from make_model import make_model
from local_feat import make_local_model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "7"


class ImageDataset(Dataset):
    """Image Dataset."""

    def __init__(self, dataset, transforms):
        _ = transforms
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        supported = [".jpg", ".JPG", ".png", ".PNG"]
        if os.path.splitext(img_path)[-1] in supported:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (opt.img_size, opt.img_size))
            img = torch.tensor(img)
            img = img[:, :, [2, 1, 0]]
            return img, img_path


def val_collate_fn(batch):
    """Val collate fn."""

    imgs, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), img_paths


# 推理
class ReID_Inference:
    """ReID Inference."""

    def __init__(self, backbone1, backbone2):

        model_backbone = {'R50': 'resnet50',
                          'RXt101ibn': 'resnext101_ibn_a',
                          'SER101ibn': 'se_resnet101_ibn_a',
                          'ResNeSt101': 'resnest101',
                          'ResNeSt269': 'resnest269'}

        # self.model1 = make_model(model_backbone[backbone1])
        # self.model2 = make_model(model_backbone[backbone2])

        self.model1 = make_local_model(model_backbone[backbone1])
        self.model2 = make_local_model(model_backbone[backbone2])

        model_name1 = opt.weights1
        model_name2 = opt.weights2

        model_path1 = os.path.join(opt.model_path, model_name1)
        model_path2 = os.path.join(opt.model_path, model_name2)

        self.model1.load_param(model_path1)
        self.model2.load_param(model_path2)

        self.model1 = nn.DataParallel(self.model1)
        self.model2 = nn.DataParallel(self.model2)

        self.batch_size = opt.batch_size
        self.model1.to('cuda')
        self.model2.to('cuda')

        self.model1.eval()
        self.model2.eval()

        self.mean = torch.tensor([123.675, 116.280, 103.530]).to('cuda')
        self.std = torch.tensor([57.0, 57.0, 57.0]).to('cuda')

    def extract(self, imgpath_list):
        """Extract feature for one image."""

        val_set = ImageDataset(imgpath_list, None)

        pin_memory = True
        num_workers = 8

        val_loader = DataLoader(
            val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=val_collate_fn,
            pin_memory=pin_memory
        )

        batch_res_dic = OrderedDict()
        for (batch_data, batch_path) in tqdm(val_loader,
                                             total=len(val_loader)):
            with torch.no_grad():
                batch_data = batch_data.to('cuda')
                batch_data = (batch_data - self.mean) / self.std
                batch_data = batch_data.permute(0, 3, 1, 2)
                batch_data = batch_data.float()
                with amp.autocast():
                    feat1 = self.model1(batch_data)
                    feat2 = self.model2(batch_data)

                feat = torch.cat((feat1, feat2), dim=1)
                feat = feat / torch.norm(feat, 2, 1, keepdim=True)
                feat = feat.cpu().detach().numpy()

            for index, imgpath in enumerate(batch_path):
                batch_res_dic[imgpath] = feat[index]
        del val_loader, val_set, feat, batch_data
        return batch_res_dic


def main():
    # 提取特征
    reid = ReID_Inference(opt.backbone1, opt.backbone2)

    info = []
    if opt.mode == 'query':
        # 提取query 特征
        for img in os.listdir(opt.root):
            img_path = os.path.join(opt.root, img)
            info.append(img_path)

        result = reid.extract(info)
        os.makedirs('outputs', exist_ok=True)
        torch.save({'features': result}, os.path.join('outputs', '{}_{}_{}_multi_features_inference.pt'.format(opt.backbone1, opt.backbone2, opt.data_code)))

    else:
        inference_info = os.listdir(opt.root)
        for cls in inference_info:
            cls_path = os.path.join(opt.root, cls)
            for img in os.listdir(cls_path):
                img_path = os.path.join(opt.root, cls, img)
                info.append(img_path)

        result = reid.extract(info)
        os.makedirs('outputs', exist_ok=True)
        # torch.save({'features': result}, os.path.join('outputs', '{}_{}_{}_multi_features.pt'.format(opt.backbone1, opt.backbone2, opt.data_code)))
        torch.save({'features': result}, os.path.join('outputs', '{}_{}_{}_multi_local.pt'.format(opt.backbone1, opt.backbone2, opt.data_code)))

    # 读取特征
    # info = torch.load(os.path.join('outputs', 'features_park_519.pt'), map_location='cuda')
    # features = info['features']
    # print(features)
    # {image_path: feature, image_path: feature, image_path: feature ......}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 超参数
    parser.add_argument('--weights1', type=str, default='resnext101_ibn_a_local.pth',
                        help='resnext101_ibn_a_100.pth'
                             'resnet50_100.pth'
                             'se_resnet101_ibn_a_100.pth'
                             'resnest101_100.pth'
                             'resnest101_2178.pth'
                             'resnest101_local_2178.pth'
                             'se_resnet101_ibn_a_2178.pth'
                             'resnest101_100_local.pth'
                             'se_resnet101_ibn_a_local.pth'
                             'resnext101_ibn_a_local.pth')

    parser.add_argument('--weights2', type=str, default='resnest101_100_local.pth',
                        help='resnext101_ibn_a_100.pth'
                             'resnet50_100.pth'
                             'se_resnet101_ibn_a_100.pth'
                             'resnest101_100.pth'
                             '')

    parser.add_argument('--backbone1', type=str, default='RXt101ibn',
                        help='R50'                  # 256
                             'RXt101ibn'            # 384
                             'SER101ibn'            # 384
                             'ResNeSt101'           # 384
                             'ResNeSt269')          # 448

    parser.add_argument('--backbone2', type=str, default='ResNeSt101',
                        help='R50'                  # 256
                             'RXt101ibn'            # 384
                             'SER101ibn'            # 384
                             'ResNeSt101'           # 384
                             'ResNeSt269')          # 448

    parser.add_argument('--img_size', type=int, default=384,
                        help='256, 384, 448, 512')

    parser.add_argument('--data_code', type=str, default='84')
    parser.add_argument('--model_path', type=str, default='../input/models/')
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--query_root', type=str, default='inference_84')
    parser.add_argument('--root', type=str, default='../input/inference_84')
    opt = parser.parse_args()

    main()

multi_model_mAP.py
import os
import sys
import numpy as np
import argparse
import time

import cv2
import torch
import torch.nn as nn
from torch.cuda import amp
from tqdm import tqdm

sys.path.append('../input/util-code/')
from make_model import make_model
from local_feat import make_local_model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

def create_model(backbone1, backbone2):
    model_backbone = {'R50': 'resnet50',
                      'RXt101ibn': 'resnext101_ibn_a',
                      'SER101ibn': 'se_resnet101_ibn_a',
                      'ResNeSt101': 'resnest101',
                      'ResNeSt269': 'resnest269'}

    # model_1 = make_model(model_backbone[backbone1])
    # model_2 = make_model(model_backbone[backbone2])

    model_1 = make_local_model(model_backbone[backbone1])
    model_2 = make_local_model(model_backbone[backbone2])

    model_name_1 = opt.weights1
    model_name_2 = opt.weights2
    model_path_1 = os.path.join(opt.model_path, model_name_1)
    model_path_2 = os.path.join(opt.model_path, model_name_2)

    model_1.load_param(model_path_1)
    model_2.load_param(model_path_2)
    model_1 = nn.DataParallel(model_1)
    model_2 = nn.DataParallel(model_2)
    model_1.to('cuda')
    model_2.to('cuda')
    model_1.eval()
    model_2.eval()
    return model_1, model_2


def load_image(img_path):
    mean = torch.tensor([123.675, 116.280, 103.530]).to('cuda')
    std = torch.tensor([57.0, 57.0, 57.0]).to('cuda')

    img = cv2.imread(img_path)
    img = cv2.resize(img, (opt.img_size, opt.img_size))
    img = torch.tensor(img)
    img = img[:, :, [2, 1, 0]]
    img = torch.unsqueeze(img, dim=0).to('cuda')
    img = (img - mean) / std
    img = img.permute(0, 3, 1, 2)
    img = img.float()
    return img


def main():
    # 读取特征
    # info = torch.load(os.path.join('outputs', '{}_{}_{}_multi_features.pt'.format(opt.backbone1, opt.backbone2, opt.data_code)), map_location='cuda')
    info = torch.load(os.path.join('outputs', '{}_{}_{}_multi_local.pt'.format(opt.backbone1, opt.backbone2, opt.data_code)), map_location='cuda')
    features = info['features']
    # {image_path: feature, image_path: feature, image_path: feature ......}

    # 提取特征
    model1, model2 = create_model(opt.backbone1, opt.backbone2)

    mAP_info = []
    top_k = opt.top_k
    all_time = []
    for i in tqdm(os.listdir(opt.root)):
        code = i.split('_')[0]
        img_path = os.path.join(opt.root, i)

        start_time = time.time()
        img = load_image(img_path)

        with amp.autocast():
            feat1 = model1(img)
            feat2 = model2(img)

        feat = torch.cat((feat1, feat2), dim=1)

        feat = feat / torch.norm(feat, 2, 1, keepdim=True)
        feat = feat.cpu().detach().numpy()
        end_time = time.time()

        result_sort = {}

        for p, f in features.items():
            score = np.dot(feat, f.T)[0] * 100
            result_sort[p] = score

        result_sort = sorted(result_sort.items(), key=lambda x: x[1], reverse=True)

        mAP_info_single = []
        count_all = 0
        count_query = 0

        for i, (k, v) in enumerate(dict(result_sort[: top_k]).items()):
            count_all += 1
            res_img_code = k.split('/')[-1].split('_')[0]

            if res_img_code == code:
                count_query += 1
                mAP_info_single.append(count_query / count_all)

        times = end_time - start_time

        all_time.append(times)
        try:
            mAP_info.append(sum(mAP_info_single) / count_query)
        except:
            mAP_info.append(sum(mAP_info_single) / 1)

    mAP = sum(mAP_info) / len(os.listdir(opt.root))
    timer = sum(all_time) / len(os.listdir(opt.root))
    print('mAP@{}: {}'.format(top_k, mAP))
    print(f'Average single img search time: {timer:.8f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 超参数
    parser.add_argument('--weights1', type=str, default='resnext101_ibn_a_local.pth',
                        help='resnext101_ibn_a_100.pth'
                             'resnet50_100.pth'
                             'se_resnet101_ibn_a_100.pth'
                             'resnest101_100.pth'
                             'resnest101_2178.pth'
                             'resnest101_local_2178.pth'
                             'se_resnet101_ibn_a_2178.pth'
                             'resnest101_100_local.pth'
                             'se_resnet101_ibn_a_local.pth'
                             'resnext101_ibn_a_local.pth')

    parser.add_argument('--weights2', type=str, default='resnest101_100_local.pth',
                        help='resnext101_ibn_a_100.pth'
                             'resnet50_100.pth'
                             'se_resnet101_ibn_a_100.pth'
                             'resnest101_100.pth'
                             '')

    parser.add_argument('--backbone1', type=str, default='RXt101ibn',
                        help='R50'                  # 256
                             'RXt101ibn'            # 384
                             'SER101ibn'            # 384
                             'ResNeSt101'           # 384
                             'ResNeSt269')          # 448

    parser.add_argument('--backbone2', type=str, default='ResNeSt101',
                        help='R50'                  # 256
                             'RXt101ibn'            # 384
                             'SER101ibn'            # 384
                             'ResNeSt101'           # 384
                             'ResNeSt269')          # 448

    parser.add_argument('--model_path', type=str, default='../input/models/')

    parser.add_argument('--img_size', type=int, default=384,
                        help='256, 384, 448, 512')

    parser.add_argument('--data_code', type=str, default='84')
    parser.add_argument('--top_k', type=int, default=29)
    parser.add_argument('--root', type=str, default='inference_84')
    opt = parser.parse_args()

    main()

