"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch_geometric
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers.base_trainer import BaseTrainer


@registry.register_trainer("mae-pair")
class EnergyTrainer(BaseTrainer):
    """
    Trainer class for the Initial Structure to Relaxed Energy (IS2RE) task.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_is2re <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_.


    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        is_hpo (bool, optional): Run hyperparameter optimization with Ray Tune.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`tensorboard`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
    """

    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        normalizer=None,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        is_hpo=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm={},
        noddp=False,
    ):
        super().__init__(
            task=task,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            identifier=identifier,
            normalizer=normalizer,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            is_hpo=is_hpo,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            name="is2re",
            slurm=slurm,
            noddp=noddp,
        )

    def load_task(self):
        logging.info(f"Loading dataset: {self.config['task']['dataset']}")
        self.num_targets = 1

    @torch.no_grad()
    def predict(
        self, loader, per_image=True, results_file=None, disable_tqdm=False
    ):
        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(loader, torch_geometric.data.Batch):
            loader = [[loader]]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        if self.normalizers is not None and "target" in self.normalizers:
            self.normalizers["target"].to(self.device)
        predictions = {"id": [], "energy": []}

        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)

            if self.normalizers is not None and "target" in self.normalizers:
                out["energy"] = self.normalizers["target"].denorm(
                    out["energy"]
                )

            if per_image:
                predictions["id"].extend(
                    [str(i) for i in batch[0].sid.tolist()]
                )
                predictions["energy"].extend(out["energy"].tolist())
            else:
                predictions["energy"] = out["energy"].detach()
                return predictions

        self.save_results(predictions, results_file, keys=["energy"])

        if self.ema:
            self.ema.restore()

        return predictions

    def train(self, disable_eval_tqdm=False):
        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        self.best_val_mae = 1e9

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)
                #import pdb; pdb.set_trace()
                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)
                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    metrics={},
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    and distutils.is_master()
                    and not self.is_hpo
                ):
                    log_str = [
                        "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    ]
                    print(", ".join(log_str))
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                # Evaluate on val set after every `eval_every` iterations.
                if self.step % eval_every == 0:
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        if (
                            val_metrics[
                                self.evaluator.task_primary_metric[self.name]
                            ]["metric"]
                            < self.best_val_mae
                        ):
                            self.best_val_mae = val_metrics[
                                self.evaluator.task_primary_metric[self.name]
                            ]["metric"]
                            self.save(
                                metrics=val_metrics,
                                checkpoint_file="best_checkpoint.pt",
                                training_state=False,
                            )
                            if self.test_loader is not None:
                                self.predict(
                                    self.test_loader,
                                    results_file="predictions",
                                    disable_tqdm=False,
                                )

                        if self.is_hpo:
                            self.hpo_update(
                                self.epoch,
                                self.step,
                                self.metrics,
                                val_metrics,
                            )

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()

    def _forward(self, batch_list):
        output = self.model(batch_list)
        #import pdb; pdb.set_trace()

        if output.shape[-1] == 1:
            output = output.view(-1)

        return {
            "energy": output,
        }

    def _compute_loss(self, out, batch_list):
        energy_target = torch.cat(
            [batch.y_relaxed.to(self.device) for batch in batch_list], dim=0
        )
        
        
        ads_label = torch.cat([batch.ads.to(self.device) for batch in batch_list], dim=0)
        bulk_label = torch.cat([batch.bulk.to(self.device) for batch in batch_list], dim=0)
        surf_label = torch.cat([batch.surf.to(self.device) for batch in batch_list], dim=0)
        #conf_label = torch.cat([batch.conf.to(self.device) for batch in batch_list], dim=0)
        
        ads_combo = torch.combinations(ads_label, r=2)
        bulk_combo = torch.combinations(bulk_label, r=2)
        surf_combo = torch.combinations(surf_label, r=2)
        key = [ads_combo[:,0] == ads_combo[:,1]] and [bulk_combo[:,0] == bulk_combo[:,1]]
        key_int = key[0].long()

        #import pdb; pdb.set_trace()
        ##################################################################
        if self.normalizer.get("normalize_labels", False):
            target_normed = self.normalizers["target"].norm(energy_target)
        else:
            target_normed = energy_target
        ##################################################################
        std = self.normalizers['target'].std
        
        target_combo = torch.combinations(target_normed, r=2)
        target_delta = target_combo[:,0] - target_combo[:,1]
        #y = torch.gt(target_delta,0).int()
        
        out_combo = torch.combinations(out["energy"], r=2)
        out_delta = out_combo[:,0] - out_combo[:,1]
        
        l = 0.4
        e = 0.2
        #key_int = 1 #only when random batch feeding
        
        # mae loss
        loss_mae = self.loss_fn["energy"](out['energy'], target_normed)
        
        # pairwise loss
        # delta
        #loss_pair = (torch.abs(target_delta-out_delta)*key_int).mean()

        # sigmoid
        loss_pair = (torch.sigmoid(e - std * out_delta * target_delta)*key_int).mean()

        
        # hinge
        # loss_pair = (torch.maximum(e - out_delta * target_delta, target_delta * 0.)*key_int).mean()
        loss_pair = torch.nan_to_num(loss_pair, nan=0)
        loss = l*loss_mae + (1-l) * loss_pair
        return loss
    
    
    
        #y_ = torch.sigmoid(out_delta)       
        #loss_BCE = (-y*torch.log(y_) - (1-y)*torch.log(1-y_)).sum() #Binary Cross Entropy
        # loss_reg =  ((torch.abs(target_delta-out_delta)/torch.tensor(1.414)).sum()
        # loss_reg = torch.abs((target_delta-out_delta)/1.414).sum() # grid search!!!!!
        
        # #self.loss_fn["energy"](out_delta, target_delta)
        #loss_rank = torch.maximum(e-(out_delta / target_delta) , target_delta * 0.).sum() # Hinge loss
        # loss_pair = (torch.maximum(e-(out_delta / target_delta) , target_delta * 0.)*key_int).sum()
        #loss_pair = (1.0 - torch.sigmoid(std* (e-out_delta / target_delta))).sum() #sigmoid
        #loss_pair = torch.sigmoid(e - std * out_delta * target_delta).mean() #sigmoid
         #+ (1.0-l) * loss_rank
        #loss = self.loss_fn["energy"](out_delta, target_delta) #L1
        # loss = (target_delta - out_delta) @ (target_delta - out_delta).t() #L2
        #loss = ((target_delta - out_delta)**2).mean() #L2
        #import pdb; pdb.set_trace()
        

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        energy_target = torch.cat(
            [batch.y_relaxed.to(self.device) for batch in batch_list], dim=0
        )
        

        if self.normalizer.get("normalize_labels", False):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])

        metrics = evaluator.eval(
            out,
            {"energy": energy_target},
            prev_metrics=metrics,
        )

        return metrics
