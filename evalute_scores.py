import numpy as np
import argparse
import torch
import os

import shutil
from pathlib import Path

from src.tools.files import json_dump
import torch.nn.functional as F
import einops

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="evaluate")
def main(cfg: DictConfig):
    fabric = instantiate(cfg.trainer.fabric)
    fabric.launch()

    for dataset in cfg.evaluate:
        columns = shutil.get_terminal_size().columns
        fabric.print(f"Evaluating {cfg.evaluate[dataset].dataname}".center(columns))

        data = instantiate(cfg.evaluate[dataset])
        test_loader = fabric.setup_dataloaders(data.test_dataloader())

        query_feats_filename = "query_feat_txt_only.pt" if cfg.evaluate[dataset].test["_target_"] == "src.test.webvid_covr_text.TestWebVidCoVRTextOnly" else "query_feat.pt"
        try:
            query_feat_file_path = os.path.join(cfg.machine.paths.output_dir, query_feats_filename)
            query_feats = torch.load(query_feat_file_path)
        except Exception as e:
            print(e)
            print("Check the query features files name and path are correct")
            return

        tar_img_feats = []
        pair_ids = []

        for _, _, tar_feat, _, pair_id, *_ in test_loader:
            pair_ids.extend(pair_id)
            tar_img_feats.append(tar_feat.cpu())

        query_feats = torch.cat(query_feats, dim=0).cpu()
        tar_img_feats = torch.cat(tar_img_feats, dim=0)

        query_feats = F.normalize(query_feats, dim=-1)
        tar_img_feats = F.normalize(tar_img_feats, dim=-1)

        ref_img_ids = [test_loader.dataset.pairid2ref[int(pair_id)] for pair_id in pair_ids]
        tar_img_ids = [test_loader.dataset.pairid2tar[int(pair_id)] for pair_id in pair_ids]

        ref_img_ids = torch.tensor(ref_img_ids, dtype=torch.long)
        tar_img_ids = torch.tensor(tar_img_ids, dtype=torch.long)

        if fabric.world_size > 1:
            # Gather tensors from every process
            query_feats = fabric.all_gather(query_feats)
            tar_img_feats = fabric.all_gather(tar_img_feats)
            ref_img_ids = fabric.all_gather(ref_img_ids)
            tar_img_ids = fabric.all_gather(tar_img_ids)

            query_feats = einops.rearrange(query_feats, "d b e -> (d b) e")
            tar_img_feats = einops.rearrange(tar_img_feats, "d b e -> (d b) e")
            ref_img_ids = einops.rearrange(ref_img_ids, "d b -> (d b)")
            tar_img_ids = einops.rearrange(tar_img_ids, "d b -> (d b)")


        sim_q2t = (query_feats @ tar_img_feats.t()).cpu().numpy()

        # Remove self similarities
        for i in range(len(ref_img_ids)):
            for j in range(len(tar_img_ids)):
                if ref_img_ids[i] == tar_img_ids[j]:
                    sim_q2t[i][j] = -10


        ranks = np.zeros(sim_q2t.shape[0])
        for index, score in enumerate(sim_q2t):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == index)[0][0]

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)  # type: ignore
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        tr50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)

        tr_mean3 = (tr1 + tr5 + tr10) / 3
        tr_mean4 = (tr1 + tr5 + tr10 + tr50) / 4

        recalls = {
            "R1": round(tr1, 2),
            "R5": round(tr5, 2),
            "R10": round(tr10, 2),
            "R50": round(tr50, 2),
            "meanR3": round(tr_mean3, 2),
            "meanR4": round(tr_mean4, 2),
        }

        recalls["annotation"] = Path(test_loader.dataset.annotation_pth).name

        suffix = "_txt_only" if cfg.evaluate[dataset].test["_target_"] == "src.test.webvid_covr_text.TestWebVidCoVRTextOnly" else ""

        print("Recalls: ")
        print(recalls)

        # Save results
        json_dump(recalls, f"recalls_covr{suffix}.json")
        print(f"Recalls saved in {Path.cwd()} as recalls_covr{suffix}.json")

if __name__ == "__main__":
    main()
