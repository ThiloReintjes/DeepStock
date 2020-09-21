import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import abc
import torch.utils.data as td
import os
import Dataset


class GeneralNetwork(pl.LightningModule):
    def __init__(self, hparams, day_size=6, num_classes=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # set hyperparams
        self.day_size = day_size
        self.input_size = day_size * hparams["days_of_data"]
        self.hparams = hparams
        self.num_classes = num_classes

        # device
        self.cur_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Cuda available:", torch.cuda.is_available())
        print("Device:", self.cur_device)

        self.datasets = {}

        # define Loss
        self.loss = self.define_loss()

    @abc.abstractmethod
    def forward(self, x, *args, **kwargs):
        pass

    def define_loss(self):
        loss = None
        if self.hparams["loss"]:
            if self.hparams["loss"] is "L1":
                loss = F.l1_loss
                print("loss is L1")
            elif self.hparams["loss"] is "MSE":
                loss = F.mse_loss
                print("loss is MSE")
        else:
            loss = F.l1_loss
            print("loss is not defined - loss is L1")
        return loss

    @abc.abstractmethod
    def compute_acc(self, out, target):
        pass

    def general_step(self, batch, train_step):
        data, target = batch

        # training or evaluating
        if train_step:
            self.train()
        else:
            self.eval()

        # forward pass
        out = self.forward(data)

        # loss
        loss = self.loss(out, target)

        # accuracy
        acc = self.handel_acc(out, target)

        return out, loss, acc

    def training_step(self, batch, batch_idx):
        out, loss, acc = self.general_step(batch, True)

        # logs
        tensorboard_logs = {'loss': loss, 'acc': acc}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        out, loss, acc = self.general_step(batch, False)

        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        out, loss, acc = self.general_step(batch, False)

        return {'test_loss': loss}

    def validation_epoch_end(self, outputs):
        # stack
        val_loss = torch.stack([x['val_loss'] for x in outputs])
        val_acc = torch.stack([x['val_acc'] for x in outputs])

        # mean
        val_loss_mean = val_loss.mean()
        val_acc_mean = val_acc.mean()

        # logs
        tensorboard_logs = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()

        # logs
        logs = {'test_loss': test_loss_mean}
        return {'test_loss': test_loss_mean, 'log': logs, 'progress_bar': logs}

    @pl.data_loader
    def train_dataloader(self):
        return td.DataLoader(self.datasets["train_dataset"], shuffle=False,
                             batch_size=self.hparams["batch_size"],
                             num_workers=10)

    @pl.data_loader
    def val_dataloader(self):
        return td.DataLoader(self.datasets["val_dataset"],
                             batch_size=self.hparams["batch_size"],
                             num_workers=5)

    @pl.data_loader
    def test_dataloader(self):
        return td.DataLoader(self.datasets["test_dataset"],
                             batch_size=self.hparams["batch_size"],
                             num_workers=5)

    def configure_optimizers(self):
        # optim
        optim = None
        if self.hparams["optim"]:
            if self.hparams["optim"] is "Adam":
                optim = torch.optim.Adam(self.parameters(), self.hparams["lr"],
                                         weight_decay=self.hparams["weight_decay"])
                print("optim is Adam")
            elif self.hparams["optim"] is "SGD":
                optim = torch.optim.SGD(self.parameters(), self.hparams["lr"],
                                        weight_decay=self.hparams["weight_decay"])
                print("optim is SGD")
        else:
            optim = torch.optim.Adam(self.parameters(), self.hparams["lr"], weight_decay=self.hparams["weight_decay"])
            print("optim is not defined - optim is Adam")

        # lr_scheduler
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=6, min_lr=1.0e-6, verbose=True,
                                                                  cooldown=10)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [7])

        return [optim], [lr_scheduler]

    def prepare_data(self):
        path = os.path.dirname(os.getcwd()) + "/Stock Datasets/Data/training"

        datasets_list = []

        for filename in os.listdir(path):
            if filename.endswith(".csv") and not "computed_" in filename:
                datasets_list.append(
                    Dataset.StockDataset(
                        csv_path=path + "/" + filename,
                        days_of_data=self.hparams["days_of_data"],
                        label=self.hparams["to_predict"],
                        label_type=self.hparams["label_type"],
                        threshold=self.hparams["threshold"],
                        normalization=self.hparams["normalize"],
                        training=True)
                )
        path = os.getcwd()

        dataset_test = Dataset.StockDataset(
            csv_path=path + "/Data/test/AAPL.csv",
            days_of_data=self.hparams["days_of_data"],
            label=self.hparams["to_predict"],
            label_type=self.hparams["label_type"],
            threshold=self.hparams["threshold"],
            normalization=self.hparams["normalize"],
            training=True,
        )

        dataset = Dataset.ConcatDataset(*datasets_list)
        # dataset = dataset_test

        length = len(dataset)
        print("Dataset length:", length)

        val_size = int(0.2 * length)
        train_size = length - val_size

        self.datasets["train_dataset"], self.datasets["val_dataset"] = td.random_split(dataset, [train_size, val_size])
        self.datasets["test_dataset"] = dataset_test

        print("train_size:", len(self.datasets["train_dataset"]))
        print("val_size:", len(self.datasets["val_dataset"]))
        print("test_size:", len(self.datasets["test_dataset"]))
