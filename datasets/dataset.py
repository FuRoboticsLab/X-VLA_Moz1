from __future__ import annotations
from typing import Dict, Iterable, List
import io, json, random, numpy as np, torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from mmengine import fileio
from .common import action_slice
from .domain_config import DATA_WEIGHTS, DATA_DOMAIN_ID
from .domain_handler.registry import get_handler_cls

class InfiniteDataReader(IterableDataset):
    """
    Output sample:
      {
        'domain_id': LongTensor[],    # domain id
        'language_instruction': str,
        'image_input': FloatTensor[V, C, H, W],
        'image_mask': BoolTensor[V],
        'proprio': FloatTensor[dim_proprio],
        'action': FloatTensor[T, dim_action]
      }
    """
    def __init__(self, 
                 metas_path: str, 
                 num_actions: int = 10, 
                 num_views: int = 3, 
                 training: bool = True,
                 lang_aug: str = None,
                 ):
        self.num_views = num_views
        self.training = training
        self.num_actions = num_actions

        self.metas: Dict[str, dict] = {}
        if fileio.isdir(metas_path):
            meta_files = fileio.list_dir_or_file(metas_path, suffix=".json", recursive=True, list_dir=False)
            root = metas_path
        else: meta_files, root = [metas_path], ""
        for file in meta_files:
            with io.BytesIO(fileio.get(fileio.join_path(root, file))) as f: meta = json.load(f)
            print(f"== dataset {meta['dataset_name']} with {len(meta['datalist'])} trajs")
            self.metas[meta["dataset_name"]] = meta

        self.image_aug = [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
        ]
        if training: self.image_aug.insert(1, transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2, hue=0.0))
        self.image_aug = transforms.Compose(self.image_aug)
        if lang_aug is not None: self.lang_aug = json.load(open(lang_aug, "r")) # support json or jsonl
        else: self.lang_aug = None

    def _iter_one_dataset(self, dataset_name: str) -> Iterable[dict]:
        meta = self.metas[dataset_name]
        traj_indices = list(range(len(meta["datalist"])))
        if self.training: random.shuffle(traj_indices)
        Handler = get_handler_cls(dataset_name)
        handler = Handler(meta=meta, num_views=self.num_views)
        for traj_idx in traj_indices:
            for sample in handler.iter_episode(
                traj_idx,
                num_actions=self.num_actions,
                training=self.training,
                image_aug=self.image_aug,
                lang_aug_map=self.lang_aug,
            ):
                sample["domain_id"] = torch.tensor(DATA_DOMAIN_ID.get(dataset_name, 0))
                rel_idx = meta.get("rel_idx", ())
                sample.update(action_slice(sample["abs_trajectory"], rel_idx))
                del sample["abs_trajectory"]
                yield sample

        if self.training: yield from self._iter_one_dataset(dataset_name)


    def __iter__(self):
        names = list(self.metas.keys())
        if not self.training: 
            for n in names: yield from self._iter_one_dataset(n)
        else:
            names = names * 20 # increase the dataset sampling frequency
            gens = [iter(self._iter_one_dataset(n)) for n in names]
            ws = [DATA_WEIGHTS.get(n, 1.0) for n in names]
            s = sum(ws); ws = [w / s for w in ws]
            while True:
                i = random.choices(range(len(names)), weights=ws, k=1)[0]
                yield next(gens[i])