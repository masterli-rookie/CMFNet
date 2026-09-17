from pathlib import Path
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    def __init__(self, txt_file, resize=(256, 256), augment=False):
        self.txt_file = txt_file
        self.resize = resize
        self.augment = augment
        self.samples = self._load_txt(txt_file)

    def _split_line(self, line):
        """
        兼容多种格式：
        1) tab 分隔：img_path \t mask_path
        2) 空格分隔：img_path mask_path
        3) 路径中带空格时，自动尝试根据“文件是否存在”推断正确切分点

        返回:
            (img_path, mask_path)
        """
        line = line.strip()
        if not line:
            return None

        # 注释行
        if line.startswith("#"):
            return None

        # 方案 1：优先按 tab 分隔
        if "\t" in line:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                img_path = parts[0].strip()
                mask_path = parts[1].strip()
                if img_path and mask_path:
                    return img_path, mask_path

        # 方案 2：如果没有 tab，尝试按“空格切分点”推断
        # 逐个尝试每一个空格位置，把它当作两列的分隔符
        # 只要左边和右边都是真实存在的文件，就认为是一个候选
        candidates = []

        for m in re.finditer(r" ", line):
            idx = m.start()
            left = line[:idx].strip()
            right = line[idx + 1 :].strip()

            if not left or not right:
                continue

            # 只要两边路径都存在，就记录为候选
            if Path(left).exists() and Path(right).exists():
                candidates.append((left, right))

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            # 如果出现多个候选，优先返回第一个
            # 一般情况下真实切分点只会有一个
            return candidates[0]

        # 方案 3：再尝试“双空格及以上”作为分隔符
        # 适合某些文本里用多个空格分列的情况
        parts = re.split(r"\s{2,}", line, maxsplit=1)
        if len(parts) == 2:
            img_path = parts[0].strip()
            mask_path = parts[1].strip()
            if img_path and mask_path:
                return img_path, mask_path

        raise ValueError(
            f"无法解析 txt 行，请检查格式：\n{line}\n"
            f"建议使用 tab 分隔：img_path\\tmask_path"
        )

    def _load_txt(self, txt_file):
        txt_file = Path(txt_file)
        assert txt_file.exists(), f"txt_file 不存在: {txt_file}"

        samples = []
        with open(txt_file, "r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                try:
                    parsed = self._split_line(line)
                    if parsed is None:
                        continue
                    img_path, mask_path = parsed
                    samples.append((img_path, mask_path))
                except Exception as e:
                    raise ValueError(
                        f"解析 txt 文件失败，行号: {line_no}\n"
                        f"内容: {line.strip()}\n"
                        f"错误: {e}"
                    )

        return samples

    def __len__(self):
        return len(self.samples)

    def _read_image(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"无法读取图像: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_mask(self, path):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"无法读取 mask: {path}")
        mask = (mask > 127).astype(np.uint8)
        return mask

    def _resize(self, img, mask):
        h, w = self.resize
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = self._read_image(img_path)
        mask = self._read_mask(mask_path)

        if self.resize is not None:
            image, mask = self._resize(image, mask)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        mask = mask.astype(np.float32)[None, ...]

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return {
            "image": image,
            "mask": mask,
            "path": img_path
        }
# 训练CRACK500 的逻辑
# from pathlib import Path
# import re

# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset


# class CrackDataset(Dataset):
#     def __init__(self, txt_file, resize=(256, 256), augment=False):
#         self.txt_file = Path(txt_file)
#         self.resize = resize
#         self.augment = augment
#         self.root_dir = self.txt_file.parent
#         self.samples = self._load_txt(self.txt_file)

#     def _resolve_path(self, p):
#         p = Path(p.strip())
#         if p.is_absolute():
#             return p
#         return (self.root_dir / p).resolve()

#     def _split_line(self, line):
#         """
#         兼容格式:
#         1) tab 分隔: img_path\tmask_path
#         2) 空格分隔: img_path mask_path
#         3) 多空格分隔
#         4) 路径中带空格时，优先尝试 tab，再尝试从左右两侧拆分
#         """
#         line = line.strip()
#         if not line:
#             return None

#         if line.startswith("#"):
#             return None

#         # 1) 优先 tab
#         if "\t" in line:
#             parts = line.split("\t", 1)
#             if len(parts) == 2:
#                 img_path = parts[0].strip()
#                 mask_path = parts[1].strip()
#                 if img_path and mask_path:
#                     return img_path, mask_path

#         # 2) 普通空白分隔
#         parts = line.split()
#         if len(parts) == 2:
#             return parts[0], parts[1]

#         # 3) 多空格分隔
#         parts = re.split(r"\s{2,}", line, maxsplit=1)
#         if len(parts) == 2:
#             img_path = parts[0].strip()
#             mask_path = parts[1].strip()
#             if img_path and mask_path:
#                 return img_path, mask_path

#         # 4) 兜底: 尝试按每个空格切分，找出左右两边都存在的情况
#         candidates = []
#         for m in re.finditer(r" ", line):
#             idx = m.start()
#             left = line[:idx].strip()
#             right = line[idx + 1 :].strip()
#             if not left or not right:
#                 continue

#             left_path = self._resolve_path(left)
#             right_path = self._resolve_path(right)
#             if left_path.exists() and right_path.exists():
#                 candidates.append((left, right))

#         if len(candidates) >= 1:
#             return candidates[0]

#         raise ValueError(
#             f"无法解析 txt 行，请检查格式：\n{line}\n"
#             f"建议格式：img_path\\tmask_path 或 img_path mask_path"
#         )

#     def _load_txt(self, txt_file):
#         txt_file = Path(txt_file)
#         assert txt_file.exists(), f"txt_file 不存在: {txt_file}"

#         samples = []
#         with open(txt_file, "r", encoding="utf-8-sig") as f:
#             for line_no, line in enumerate(f, start=1):
#                 try:
#                     parsed = self._split_line(line)
#                     if parsed is None:
#                         continue
#                     img_path, mask_path = parsed
#                     img_path = self._resolve_path(img_path)
#                     mask_path = self._resolve_path(mask_path)
#                     samples.append((img_path, mask_path))
#                 except Exception as e:
#                     raise ValueError(
#                         f"解析 txt 文件失败，行号: {line_no}\n"
#                         f"内容: {line.strip()}\n"
#                         f"错误: {e}"
#                     )

#         return samples

#     def __len__(self):
#         return len(self.samples)

#     def _read_image(self, path):
#         img = cv2.imread(str(path), cv2.IMREAD_COLOR)
#         if img is None:
#             raise RuntimeError(f"无法读取图像: {path}")
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         return img

#     def _read_mask(self, path):
#         mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             raise RuntimeError(f"无法读取 mask: {path}")
#         mask = (mask > 127).astype(np.uint8)
#         return mask

#     def _resize(self, img, mask):
#         h, w = self.resize
#         img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
#         mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
#         return img, mask

#     def __getitem__(self, idx):
#         img_path, mask_path = self.samples[idx]

#         image = self._read_image(img_path)
#         mask = self._read_mask(mask_path)

#         if self.resize is not None:
#             image, mask = self._resize(image, mask)

#         image = image.astype(np.float32) / 255.0
#         image = np.transpose(image, (2, 0, 1))
#         mask = mask.astype(np.float32)[None, ...]

#         return {
#             "image": torch.from_numpy(image).float(),
#             "mask": torch.from_numpy(mask).float(),
#             "path": str(img_path)
#         }


###################  兼容版逻辑
# from pathlib import Path
# import re
#
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
#
#
# class CrackDataset(Dataset):
#     def __init__(self, txt_file, resize=(256, 256), augment=False):
#         self.txt_file = Path(txt_file).resolve()
#         self.resize = resize
#         self.augment = augment
#         self.root_dir = self.txt_file.parent
#         self.samples = self._load_txt(self.txt_file)
#
#     def _try_resolve(self, p):
#         """
#         尝试把 txt 中的路径解析成真实存在的文件路径：
#         1) 原始绝对路径
#         2) 相对 txt 文件所在目录
#         3) 原样追加到当前工作目录
#         """
#         p = str(p).strip()
#         if not p:
#             return None
#
#         path = Path(p)
#
#         # 1. 如果是绝对路径，直接试
#         if path.is_absolute():
#             if path.exists():
#                 return path
#             return path
#
#         # 2. 相对 txt 所在目录
#         cand1 = (self.root_dir / path).resolve()
#         if cand1.exists():
#             return cand1
#
#         # 3. 相对当前工作目录
#         cand2 = path.resolve()
#         if cand2.exists():
#             return cand2
#
#         # 4. 都不存在，返回最可能的那个
#         return cand1
#
#     def _split_line(self, line):
#         """
#         支持：
#         1) tab 分隔
#         2) 空格分隔
#         3) 多空格分隔
#         4) 路径里有空格时，通过尝试存在性来判断切分
#         """
#         line = line.strip()
#         if not line or line.startswith("#"):
#             return None
#
#         # 1) tab
#         if "\t" in line:
#             parts = line.split("\t", 1)
#             if len(parts) == 2:
#                 return parts[0].strip(), parts[1].strip()
#
#         # 2) 普通 split
#         parts = line.split()
#         if len(parts) == 2:
#             return parts[0].strip(), parts[1].strip()
#
#         # 3) 多空格 split
#         parts = re.split(r"\s{2,}", line, maxsplit=1)
#         if len(parts) == 2:
#             return parts[0].strip(), parts[1].strip()
#
#         # 4) 逐个空格试切分
#         candidates = []
#         for m in re.finditer(r" ", line):
#             idx = m.start()
#             left = line[:idx].strip()
#             right = line[idx + 1:].strip()
#             if not left or not right:
#                 continue
#
#             left_path = self._try_resolve(left)
#             right_path = self._try_resolve(right)
#
#             if left_path is not None and right_path is not None:
#                 if Path(left_path).exists() and Path(right_path).exists():
#                     candidates.append((left, right))
#
#         if candidates:
#             return candidates[0]
#
#         raise ValueError(
#             f"无法解析 txt 行:\n{line}\n"
#             f"建议格式: img_path\\tmask_path"
#         )
#
#     def _load_txt(self, txt_file):
#         assert txt_file.exists(), f"txt_file 不存在: {txt_file}"
#
#         samples = []
#         with open(txt_file, "r", encoding="utf-8-sig") as f:
#             for line_no, line in enumerate(f, start=1):
#                 parsed = self._split_line(line)
#                 if parsed is None:
#                     continue
#
#                 img_path, mask_path = parsed
#                 img_path = self._try_resolve(img_path)
#                 mask_path = self._try_resolve(mask_path)
#
#                 samples.append((img_path, mask_path, line_no, line.strip()))
#
#         print(f"Loaded {len(samples)} samples from {txt_file}")
#         return samples
#
#     def __len__(self):
#         return len(self.samples)
#
#     def _read_image(self, path):
#         path = Path(path)
#         if not path.exists():
#             raise RuntimeError(f"图像文件不存在: {path}")
#
#         img = cv2.imread(str(path), cv2.IMREAD_COLOR)
#         if img is None:
#             raise RuntimeError(f"OpenCV 无法读取图像: {path}")
#
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         return img
#
#     def _read_mask(self, path):
#         path = Path(path)
#         if not path.exists():
#             raise RuntimeError(f"mask 文件不存在: {path}")
#
#         mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             raise RuntimeError(f"OpenCV 无法读取 mask: {path}")
#
#         mask = (mask > 127).astype(np.uint8)
#         return mask
#
#     def _resize(self, img, mask):
#         h, w = self.resize
#         img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
#         mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
#         return img, mask
#
#     def __getitem__(self, idx):
#         img_path, mask_path, line_no, raw_line = self.samples[idx]
#
#         # 最终强校验
#         if not Path(img_path).exists():
#             raise RuntimeError(
#                 f"第 {line_no} 行图像不存在:\n"
#                 f"txt内容: {raw_line}\n"
#                 f"解析后图像路径: {img_path}"
#             )
#
#         if not Path(mask_path).exists():
#             raise RuntimeError(
#                 f"第 {line_no} 行mask不存在:\n"
#                 f"txt内容: {raw_line}\n"
#                 f"解析后mask路径: {mask_path}"
#             )
#
#         image = self._read_image(img_path)
#         mask = self._read_mask(mask_path)
#
#         if self.resize is not None:
#             image, mask = self._resize(image, mask)
#
#         image = image.astype(np.float32) / 255.0
#         image = np.transpose(image, (2, 0, 1))
#         mask = mask.astype(np.float32)[None, ...]
#
#         return {
#             "image": torch.from_numpy(image).float(),
#             "mask": torch.from_numpy(mask).float(),
#             "path": str(img_path)
#         }
