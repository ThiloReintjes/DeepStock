from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np


class StockDataset(Dataset):
    def __init__(self, csv_path, days_of_data, label, label_type, threshold, normalization, training):
        self.csv_path = csv_path
        self.csv_file = None
        self.days_of_data = days_of_data
        self.label = label
        self.label_type = label_type
        self.threshold = threshold
        self.normalization = normalization
        self.training = training
        self.computed_name = None
        self.dataset = None

        # check if already computed
        name = os.path.basename(os.path.normpath(self.csv_path))

        self.computed_name = "computed_" + self.label + "_" + self.label_type + "_" + str(
            self.threshold) + "_" + self.normalization + name
        computed_path = os.path.dirname(self.csv_path) + "/" + self.computed_name

        if os.path.isfile(computed_path) and training:
            print("loading", self.computed_name)
            self.dataset = pd.read_csv(computed_path, sep=",")

        elif "computed" not in name:
            print("computing", self.computed_name, "...")
            self.csv_file = pd.read_csv(csv_path, sep=",")
            self.dataset = self.compute_data()
            print("computing finished.")

    def __len__(self):
        return len(self.dataset) - self.days_of_data

    def __getitem__(self, index):
        rows = self.dataset.iloc[index:index + self.days_of_data]

        label = rows.iat[-1, -1]
        label = torch.reshape(torch.tensor(label, dtype=torch.float), shape=(-1,))

        extra_data = None

        data = rows.drop(rows.columns[-1], axis=1)

        if not self.training:
            temp = index + self.days_of_data
            # temp save Date and original Label
            date = self.dataset["Date"].iloc[temp - 1:temp + 1]
            close = self.dataset[self.label].iloc[temp - 1:temp + 1]
            label_unnorm = label
            data = data.drop('Date', axis=1)
            norm_info = (None, None)
            extra_data = [date, close, norm_info, self.csv_path, label_unnorm]

        data, label, extra_data = self.compute_norm(data, label, extra_data)

        data = torch.tensor(data, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)

        if not self.training:
            return data, label, extra_data
        return data, label

    def compute_data(self):
        data = self.csv_file
        data = data.fillna(method='ffill')

        # add volatility
        # TODO

        # add Change, Gain and Loss
        data["Change"] = 0
        data["Gain"] = 0
        data["Loss"] = 0
        for i in range(len(data)):
            if i > 1:
                change_i = data.loc[i, "Close"] - data.loc[i - 1, "Close"]
                data.loc[i, "Change"] = change_i
                if change_i > 0:
                    data.loc[i, "Gain"] = change_i
                else:
                    data.loc[i, "Loss"] = abs(change_i)
        data = data.drop(data.index[[0]])
        data = data.reset_index(drop=True)

        # add RSI 14 - Close - SMA
        data["avg_gain"] = data["Gain"].rolling(window=14).mean()
        data["avg_loss"] = data["Loss"].rolling(window=14).mean()
        data["RSI"] = 0
        for i in range(len(data)):
            if i > 13:
                if data.loc[i, "avg_loss"] == 0:
                    rsi = 100
                elif data.loc[i, "avg_gain"] == 0:
                    rsi = 0
                else:
                    rs = data.loc[i, "avg_gain"] / data.loc[i, "avg_loss"]
                    rsi = 100 - (100 / (1 + rs))
                data.loc[i, "RSI"] = rsi

        data = data.drop(data.index[:14])
        data = data.reset_index(drop=True)

        # add Label - Close % of change
        data = self.compute_label(data)

        # drop unwanted columns
        if self.training:
            data = data.drop('Date', axis=1)
        data = data.drop('Adj Close', axis=1)
        data = data.drop('Change', axis=1)
        data = data.drop('Gain', axis=1)
        data = data.drop('Loss', axis=1)
        data = data.drop('avg_gain', axis=1)
        data = data.drop('avg_loss', axis=1)

        # assert not data.isnull().values.any()
        data = data.fillna(method='ffill')

        if self.training:
            computed_path = os.path.dirname(self.csv_path) + "/" + self.computed_name
            data.to_csv(computed_path, index=False, index_label=False)

        return data

    def compute_label(self, data):
        if self.label_type is "percent":
            return self.label_percent(data)

        if self.label_type is "real":
            return self.label_real(data)

        if self.label_type is "classification":
            return self.label_classification(data)

        raise Exception("No label computed!")

    def label_percent(self, data):
        data["Label"] = 0
        for i in range(len(data)):
            label = None
            # % of change
            if i < len(data) - 1:
                close0 = data.loc[i, "Close"]
                close1 = data.loc[i + 1, self.label]
                label = close1 / close0
                data.loc[i, "Label"] = label
        data = data.drop(data.tail(1).index)
        data = data.reset_index(drop=True)
        return data

    def label_real(self, data):
        data["Label"] = 0
        for i in range(len(data)):
            label = None
            # close of next day
            if i < len(data) - 1:
                close1 = data.loc[i + 1, self.label]
                data.loc[i, "Label"] = close1
        data = data.drop(data.tail(1).index)
        data = data.reset_index(drop=True)
        return data

    def label_classification(self, data):
        data["Label"] = 1
        for i in range(len(data)):
            label = None
            # 0 loss, 1 neutral, 2 gain
            if i < len(data) - 1:
                close0 = data.loc[i, self.label]
                close1 = data.loc[i + 1, self.label]
                percent = close1 / close0 - 1
                if percent > self.threshold:
                    data.loc[i, "Label"] = 2
                if percent < -self.threshold:
                    data.loc[i, "Label"] = 0
        data = data.drop(data.tail(1).index)
        data = data.reset_index(drop=True)
        return data

    def compute_norm(self, data, label, extra_data):
        if self.normalization is "min_max":
            return self.norm_min_max(data, label, extra_data)

        if self.normalization is "decimal":
            return self.norm_decimal(data, label, extra_data)

        if self.normalization is "None":
            return self.norm_none(data, label, extra_data)

        raise Exception("No label computed!")

    def norm_none(self, data, label, extra_data):
        # to_numpy
        data = data.to_numpy(dtype=float, copy=True)

        return data, label, extra_data

    def norm_min_max(self, data, label, extra_data):
        # separate RSI and Volume
        rsi = data["RSI"]
        vol = data["Volume"]
        data = data.drop('RSI', axis=1)
        data = data.drop('Volume', axis=1)

        # to_numpy
        data = data.to_numpy(dtype=float, copy=True)
        rsi = rsi.to_numpy(dtype=float, copy=True) * 0.01
        rsi = rsi.reshape(1, -1)
        vol = vol.to_numpy(dtype=float, copy=True)
        vol = vol.reshape(1, -1)

        # normalize column wise
        data_min = data.min()
        data_max = data.max()
        top = data - data_min
        bottom = data_max - data_min
        data = 2 * (top / bottom) - 1

        # normalize Volume
        vol_norm = 2 * ((vol - vol.min()) / (vol.max() - vol.min())) - 1

        # append Volume and RSI
        data = np.append(data, np.transpose(vol_norm), axis=1)
        data = np.append(data, np.transpose(rsi), axis=1)

        if self.label_type is "real":
            label = 2 * ((label - data_min) / (data_max - data_min)) - 1
            if not self.training:
                extra_data[2] = (data_min, data_max)

        return data, label, extra_data

    def norm_decimal(self, data, label, extra_data):
        # TODO
        return data, label, extra_data


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        length = 0
        for d in self.datasets:
            if i - length < len(d):
                data = d.__getitem__(i - length)
                return data
            else:
                length += len(d)

    def __len__(self):
        return sum(len(d) for d in self.datasets)
