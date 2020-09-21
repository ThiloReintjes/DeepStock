import torch


def last_n_days(model, dataset, n, pred_type, thresholde=0.05):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pred = []
    long = []
    short = []

    for i in range(n):
        with torch.no_grad():
            data, label, saved_data = dataset.__getitem__(len(dataset) - n + i - 1)

            # to(device)
            data = data.to(device)
            label = label.to(device)

            # compute
            model.eval()
            data = data.unsqueeze(0)
            y_hat = model.forward(data).item()

            if pred_type is "classification":
                if y_hat is 2 or 0:
                    # calculate
                    close0 = saved_data[1].iloc[0].item()
                    close1 = saved_data[4].item()
                    diff = close1 - close0
                    percent = (close1 / close0 - 1) * 100

                    buy_date = saved_data[0].iloc[0]
                    if y_hat is 2:
                        long.append({"buy_date": buy_date, "diff": diff, "percent": percent, "predicted": y_hat,
                                     "label": label[0].item()})
                    elif y_hat is 0:
                        short.append({"buy_date": buy_date, "diff": diff, "percent": percent, "predicted": y_hat,
                                      "label": label[0].item()})

            if pred_type is "percent":
                if 0 < y_hat - 1 >= thresholde or 0 > y_hat - 1 <= thresholde * -1:
                    # calculate
                    close0 = saved_data[1].iloc[0].item()
                    close1 = saved_data[4].item()
                    diff = close1 - close0
                    percent = (close1 / close0 - 1)

                    buy_date = saved_data[0].iloc[0]
                    if 0 < y_hat - 1 >= thresholde:
                        long.append({"buy_date": buy_date, "diff": diff, "percent": percent, "predicted": y_hat,
                                     "label": label[0].item()})

                    elif 0 > y_hat - 1 <= thresholde * -1:
                        short.append({"buy_date": buy_date, "diff": diff, "percent": percent, "predicted": y_hat,
                                      "label": label[0].item()})

            if pred_type is "real":
                y_hat = ((y_hat + 1) / 2) * (saved_data[2][1] - saved_data[2][0]) + saved_data[2][0]
                close_percent = y_hat / saved_data[1].iloc[0]
                if close_percent >= 1 + thresholde or close_percent <= 1 - thresholde:
                    # calculate
                    close0 = saved_data[1].iloc[0].item()
                    close1 = saved_data[4].item()
                    diff = close1 - close0
                    percent = (close1 / close0 - 1)

                    buy_date = saved_data[0].iloc[0]
                    if close_percent >= 1 + thresholde:
                        long.append({"buy_date": buy_date, "diff": diff, "percent": percent, "predicted": y_hat,
                                     "label": label})
                    elif close_percent <= 1 - thresholde:
                        short.append({"buy_date": buy_date, "diff": diff, "percent": percent, "predicted": y_hat,
                                      "label": label})

            pred.append({"predicted": y_hat, "label": label[0].item()})

        return pred, long, short
