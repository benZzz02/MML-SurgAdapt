import glob
import json
import os
import random
from copy import deepcopy

import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from torchvision import datasets as datasets

from config import cfg
from log import logger


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0  # type: ignore
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


class AverageMeter(object):

    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01  # type: ignore


def _downsample_curve(precision, recall, max_points=20000):
    precision = np.asarray(precision)
    recall = np.asarray(recall)
    n = len(precision)
    if n <= max_points:
        return precision, recall
    idx = np.linspace(0, n - 1, max_points).astype(np.int64)
    return precision[idx], recall[idx]


def _pr_curve_safe(y_true, y_score, max_points=20000):
    """
    sklearn 在没有正样本时会 warning/不稳定，这里兜底：
    - 没有正样本：返回退化曲线 (precision=1, recall=0)
    """
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float32)

    if y_true.sum() == 0:
        precision = np.array([1.0], dtype=np.float32)
        recall = np.array([0.0], dtype=np.float32)
        return precision, recall

    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    precision, recall = _downsample_curve(precision, recall, max_points=max_points)
    return precision.astype(np.float32), recall.astype(np.float32)


def recall_at_fixed_precision(y_true, y_score, p_target=0.90):
    """
    从 PR 曲线上读点：max recall s.t. precision >= p_target
    返回 [0,1] recall
    """
    precision, recall = _pr_curve_safe(y_true, y_score, max_points=10**9)
    mask = precision >= float(p_target)
    return float(np.max(recall[mask])) if np.any(mask) else 0.0


def micro_recall_at_fixed_precision(Y_true, Y_score, p_target=0.90):
    return recall_at_fixed_precision(np.ravel(Y_true), np.ravel(Y_score), p_target=p_target)


def compute_pr_sidecar(labels, preds, video_ids, pr_targets=(0.70, 0.80, 0.90), max_points=20000):
    """
    三个数据集都用 micro PR（各一条曲线）：
    - 产出 micro 曲线
    - 产出 R@Pxx_micro（固定 precision 的 recall）
    """
    out = {
        "PR_targets": [float(p) for p in pr_targets],
        "Cholec80": {},
        "Endoscapes": {},
        "CholecT50": {},
    }

    def build_micro(y, s):
        y_flat = np.ravel(y).astype(np.int32)
        s_flat = np.ravel(s).astype(np.float32)

        prec, rec = _pr_curve_safe(y_flat, s_flat, max_points=max_points)

        points_micro = {
            f"R@P{int(p*100)}_micro": micro_recall_at_fixed_precision(y, s, p_target=p) * 100.0
            for p in pr_targets
        }

        return {
            "curve_micro": {"precision": prec, "recall": rec},
            "points_micro": points_micro,
        }

    video_ids = np.asarray(video_ids)

    mask = np.array(["cholec80" in str(v) for v in video_ids])
    if np.any(mask):
        y = labels[mask][:, :7]
        s = preds[mask][:, :7]
        out["Cholec80"] = build_micro(y, s)

    mask = np.array(["endoscapes" in str(v) for v in video_ids])
    if np.any(mask):
        y = labels[mask][:, 7:10]
        s = preds[mask][:, 7:10]
        out["Endoscapes"] = build_micro(y, s)

    mask = np.array(["cholect50" in str(v) for v in video_ids])
    if np.any(mask):
        y = labels[mask][:, 10:]
        s = preds[mask][:, 10:]
        out["CholecT50"] = build_micro(y, s)

    return out


def _find_latest_result_json(results_dir, test=True):
    """
    自动找到 results/{dir}/ 下最新的 result*.json，用它的 basename 作为 PR 文件前缀
    """
    folder = f"results/{results_dir}"
    pattern = os.path.join(folder, "result_*_test.json") if test else os.path.join(folder, "result_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def merge_pr_points_into_result_json(pr_data, results_dir, test=True, result_json_path=None):
    """
    把 R@Pxx_micro 写回 result.json：
      data[ds]["PR_micro_points"] = {...}   # 只包含 R@Pxx_micro
    同时写 meta：
      data["PR_meta"] = {"PR_targets": [...]}
    """
    folder = f"results/{results_dir}"
    os.makedirs(folder, exist_ok=True)

    path = result_json_path or _find_latest_result_json(results_dir, test=test)
    if not path:
        print("[PR] result.json not found, skip merge.")
        return None

    with open(path, "r") as f:
        data = json.load(f)

    for ds in ["Cholec80", "Endoscapes", "CholecT50"]:
        if ds in data and ds in pr_data and "points_micro" in pr_data[ds]:
            data[ds]["PR_micro_points"] = pr_data[ds]["points_micro"]

    data["PR_meta"] = {"PR_targets": pr_data.get("PR_targets", [])}

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[PR] merged points into: {path}")
    return path


def save_pr_npz_and_png(pr_data, results_dir, ckpt_path=None, test=True, result_json_path=None):
    """
    命名：跟 result.json 前缀一致（basename 去掉 .json）
      - {prefix}_pr.npz
      - {prefix}_pr_cholec80_micro.png
      - {prefix}_pr_endoscapes_micro.png
      - {prefix}_pr_cholect50_micro.png
    """
    folder = f"results/{results_dir}"
    os.makedirs(folder, exist_ok=True)

    res_path = result_json_path or _find_latest_result_json(results_dir, test=test)

    if res_path:
        prefix = os.path.splitext(os.path.basename(res_path))[0]
    else:
        from datetime import datetime
        tag = os.path.splitext(os.path.basename(ckpt_path))[0] if ckpt_path else "unknown"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"result_{ts}_{tag}" + ("_test" if test else "")

    npz_path = os.path.join(folder, f"{prefix}_pr.npz")
    arrays = {}

    def pack_micro(ds_key, prefix_key):
        if ds_key not in pr_data or "curve_micro" not in pr_data[ds_key]:
            return
        c = pr_data[ds_key]["curve_micro"]
        arrays[f"{prefix_key}_micro_precision"] = c["precision"]
        arrays[f"{prefix_key}_micro_recall"] = c["recall"]
        for k, v in pr_data[ds_key]["points_micro"].items():
            arrays[f"{prefix_key}_{k}"] = np.array([v], dtype=np.float32)

    pack_micro("Cholec80", "cholec80")
    pack_micro("Endoscapes", "endoscapes")
    pack_micro("CholecT50", "cholect50")

    np.savez_compressed(npz_path, **arrays)
    print(f"[PR] saved npz: {npz_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag_for_title = os.path.splitext(os.path.basename(ckpt_path))[0] if ckpt_path else prefix

    def plot_one(ds_key, title, fname_key):
        if ds_key not in pr_data or "curve_micro" not in pr_data[ds_key]:
            return
        d = pr_data[ds_key]["curve_micro"]
        plt.figure()
        plt.plot(d["recall"], d["precision"], label="micro")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{title} micro PR ({tag_for_title})")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        png_path = os.path.join(folder, f"{prefix}_pr_{fname_key}_micro.png")
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[PR] saved png: {png_path}")

    plot_one("Cholec80", "Cholec80", "cholec80")
    plot_one("Endoscapes", "Endoscapes", "endoscapes")
    plot_one("CholecT50", "CholecT50", "cholect50")

    return {
        "npz_path": npz_path,
        "prefix": prefix,
        "results_dir": folder,
    }


def compute_merge_and_save_pr(
    labels,
    preds,
    video_ids,
    results_dir,
    ckpt_path=None,
    test=True,
    pr_targets=(0.70, 0.80, 0.90),
    max_points=20000,
):
    """
    一键：
      1) compute_pr_sidecar
      2) merge_pr_points_into_result_json
      3) save_pr_npz_and_png
    """
    pr_data = compute_pr_sidecar(labels, preds, video_ids, pr_targets=pr_targets, max_points=max_points)
    merged_path = merge_pr_points_into_result_json(pr_data, results_dir, test=test)
    save_info = save_pr_npz_and_png(pr_data, results_dir, ckpt_path=ckpt_path, test=test, result_json_path=merged_path)
    return pr_data, merged_path, save_info


class CocoDetection(datasets.coco.CocoDetection):

    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)

    def labels(self):
        return [v["name"] for v in self.coco.cats.values()]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:  # type: ignore
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']  # type: ignore
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = target.max(dim=0)[0]
        return img, target


class COCO_missing_dataset(torch.utils.data.Dataset):  # type: ignore

    def __init__(self,
                 root,
                 annFile,
                 transform=None,
                 target_transform=None,
                 class_num: int = -1):
        self.root = root
        with open(annFile, 'r') as f:
            names = f.readlines()
        # name = names.strip('\n').split(' ')
        self.name = names
        # self.label = name[:,1]
        self.transform = transform
        self.class_num = class_num
        self.target_transform = target_transform

    def __getitem__(self, index):
        name = self.name[index]
        path = name.strip('\n').split(',')[0]
        num = name.strip('\n').split(',')[1]
        num = num.strip(' ').split(' ')
        num = np.array([int(i) for i in num])
        label = np.zeros([self.class_num])
        label[num] = 1
        label = torch.tensor(label, dtype=torch.long)
        if os.path.exists(os.path.join(self.root, path)) == False:
            label = np.zeros([self.class_num])
            label = torch.tensor(label, dtype=torch.long)
            img = np.zeros((448, 448, 3))
            img = Image.fromarray(np.uint8(img))  # type: ignore
            exit(1)
        else:
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)  # type: ignore # noqa
        assert (self.target_transform is None)
        return [index,img], label

    def __len__(self):
        return len(self.name)

    def labels(self):
        if "coco" in cfg.data:
            assert (False)
        elif "nuswide" in cfg.data:
            with open('nuswide_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        elif "voc" in cfg.data:
            with open('voc_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        elif "cub" in cfg.data:
            with open('cub_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        else:
            assert (False)


class COCO_missing_val_dataset(torch.utils.data.Dataset):  # type: ignore

    def __init__(self,
                 root,
                 annFile,
                 transform=None,
                 target_transform=None,
                 class_num: int = -1):
        self.root = root
        with open(annFile, 'r') as f:
            names = f.readlines()
        # name = names.strip('\n').split(' ')
        self.name = names
        # self.label = name[:,1]
        self.transform = transform
        self.class_num = class_num
        self.target_transform = target_transform

    def __getitem__(self, index):
        name = self.name[index]
        path = name.strip('\n').split(',')[0]
        num = name.strip('\n').split(',')[1]
        num = num.strip(' ').split(' ')
        num = np.array([int(i) for i in num])
        label = np.zeros([self.class_num])
        label[num] = 1
        label = torch.tensor(label, dtype=torch.long)
        if os.path.exists(os.path.join(self.root, path)) == False:
            label = np.zeros([self.class_num])
            label = torch.tensor(label, dtype=torch.long)
            img = np.zeros((448, 448, 3))
            img = Image.fromarray(np.uint8(img))  # type: ignore
            exit(1)
        else:
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)  # type: ignore # noqa
        assert (self.target_transform is None)
        return img, label

    def __len__(self):
        return len(self.name)

    def labels(self):
        if "coco" in cfg.data:
            assert (False)
        elif "nuswide" in cfg.data:
            with open('nuswide_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        elif "voc" in cfg.data:
            with open('voc_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        elif "cub" in cfg.data:
            with open('cub_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        else:
            assert (False)


class ModelEma(torch.nn.Module):

    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(),
                                      model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model,
                     update_fn=lambda e, m: self.decay * e +
                     (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):

    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255),
                      random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)  # type: ignore

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    gcn = []
    gcn_no_decay = []
    prefix = "module." if torch.cuda.device_count() > 1 else "" 
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if name.startswith(f"{prefix}gc"):
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                gcn_no_decay.append(param)
            else:
                gcn.append(param)
            assert("gcn" in cfg.model_name)
        elif len(param.shape) == 1 or name.endswith(
                ".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{
        'params': no_decay,
        'weight_decay': 0.
    }, {
        'params': decay,
        'weight_decay': weight_decay
    }, {
        'params': gcn_no_decay,
        'weight_decay': 0.
    }, {
        'params': gcn,
        'weight_decay': weight_decay
    }]

def get_ema_co():
    if "coco" in cfg.data:
        ema_co = np.exp(np.log(0.82)/(641*cfg.ratio))  # type: ignore
        # ema_co = 0.9997
    elif "nus" in cfg.data:
        ema_co = np.exp(np.log(0.82)/(931*cfg.ratio))  # type: ignore
        # ema_co = 0.9998
    elif "voc" in cfg.data:
        ema_co = np.exp(np.log(0.82)/(45*cfg.ratio))  # type: ignore
        # ema_co = 0.9956
    elif "cub" in cfg.data:
        if cfg.batch_size == 96:
            ema_co = np.exp(np.log(0.82)/(63*cfg.ratio))
        else:
            ema_co = np.exp(np.log(0.82)/(47*cfg.ratio))  # type: ignore
    elif "cholec" in cfg.data:
        ema_co = np.exp(np.log(0.82)/(641*cfg.ratio))
    else:
        assert(False)
    return ema_co
