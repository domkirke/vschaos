import torch, gc
import matplotlib.pyplot as plt
from .train_train import SimpleTrainer
from ..utils.misc import apply_method


class PredictiveTrainer(SimpleTrainer):
    def __init__(self, *args, sequence_length=None, prediction_length=None, cpc=False, teacher_rate = 1., **kwargs):
        super(PredictiveTrainer, self).__init__(*args, **kwargs)
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.cpc = cpc
        self.teacher_rate = teacher_rate


    def run(self, loader, preprocessing=None, epoch=None, optimize=True, n_predictions=3, **kwargs):
        # train phase
        batch = 0; current_loss = 0;
        train_losses = None;
        if preprocessing is None:
            preprocessing = self.preprocessing
        for x,y in loader:
            # forward
            if preprocessing is not None:
                x = preprocessing(x)
            out = self.models.forward(x, y=y, n_preds=n_predictions)
            # forward negative examples in case
            out_negative = None; true_ids = None
            if kwargs.get('cpc'):
                false_targets, true_ids = loader.get_cpc_examples(x, preprocessing, prob=0.5)
                if preprocessing is not None:
                    false_targets = preprocessing(false_targets)
                out_negative = self.models.forward(false_targets, n_preds=0)
            # compute loss
            batch_loss, losses = self.losses(model=self.models, out=out, target=x, true_ids=true_ids,
                                                  out_negative=out_negative, epoch=epoch, n_preds=n_predictions)
            train_losses = losses if train_losses is None else train_losses + losses
            # learn
            if optimize:
                self.optimize(self.models, batch_loss)
            # update loop
            print("epoch %d / batch %d \nlosses : %s "%(epoch, batch, self.losses.get_named_losses(losses)))
            current_loss += float(batch_loss)
            batch += 1
            del out; del x
            gc.collect(); gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        current_loss /= batch
        return current_loss, losses

    def train(self, partition=None, write=False, batch_size=64, tasks=None, batch_cache_size=1, **kwargs):
        apply_method(self.models, 'train')
        if partition is None:
            partition = 'train'
        loader = self.get_dataloader(self.datasets, batch_size=batch_size, partition=partition, tasks=tasks, batch_cache_size=batch_cache_size)
        full_loss, full_losses = self.run(loader, n_predictions=self.prediction_length, cpc=self.cpc, **kwargs)
        if write:
            apply_method(self.losses, 'write', 'train', full_losses)
        return full_loss, full_losses

    def test(self, partition=None, write=False, tasks=None, batch_size=None, batch_cache_size=1, **kwargs):
        apply_method(self.models, 'eval')
        if batch_size is None:
            batch_size=min(partition_size, 1000)
        if partition is None:
            partition = 'test'
        with torch.no_grad():
            partition_size = len(self.datasets.partitions['test'])
            loader = self.get_dataloader(self.datasets, batch_size=batch_size, tasks=tasks, batch_cache_size=batch_cache_size, partition=partition)
            full_loss, full_losses = self.run(loader, optimize=False, n_predictions=self.prediction_length, cpc=self.cpc, **kwargs)
        if write:
            apply_method(self.losses, 'write', 'test', full_losses)
        # check if current model is the best model
        if full_loss < self.best_test_loss:
            models = self.models
            self.best_model = self.get_best_model()
        return full_loss, full_losses

    def get_dataloader(self, datasets, batch_size=64, partition=None, tasks=None, batch_catch_size=1, **kwargs):
        return self.dataloader_class(datasets, batch_size=batch_size, tasks=None, batch_cache_size=1, partition=partition)

