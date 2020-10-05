import unittest
from src.Dataset import StockDataset


class TestDataloader(unittest.TestCase):

    def test_label(self):
        dataset = StockDataset("TestData.csv", 4, "Close", "percent", 0, "None", False)
        data, label, extra_data = dataset.__getitem__(0)
        gorund = 130 / 140
        self.assertEqual(gorund, label)

        dataset = StockDataset("TestData.csv", 4, "Close", "real", 0, "None", False)
        data, label, extra_data = dataset.__getitem__(0)
        gorund = 130
        self.assertEqual(gorund, label)

        dataset = StockDataset("TestData.csv", 4, "Close", "classification", 0.05, "None", False)
        data, label, extra_data = dataset.__getitem__(0)
        gorund = 0
        self.assertEqual(gorund, label)

    def test_norm(self):
        dataset = StockDataset("TestData.csv", 4, "Close", "real", 0, "min_max", False)
        data, label, extra_data = dataset.__getitem__(0)
        # print(data)
        ground = [[100, 110, 90, 100, 101, 300000],
                  [110, 150, 110, 140, 141, 250000],
                  [100, 110, 90, 100, 101, 250000],
                  [110, 150, 110, 140, 141, 260000]]

        self.assertEqual(1, data[1][1])
        self.assertEqual(-1, data[2][2])

        self.assertEqual(1, data[0][4])
        self.assertEqual(-1, data[1][4])

    def test_index(self):
        dataset = StockDataset("TestData.csv", 14, "Close", "percent", 0, "min_max", False)
        dataset.__getitem__(0)
        last = len(dataset)
        dataset.__getitem__(last - 1)
