import torch
import rei504m_dataset

dataset = rei504m_dataset.TreesDataset(json_path='./Trees.json', image_root='./')

train_loader, val_loader, test_loader = rei504m_dataset.get_dataloaders(dataset, batch_size=30, val_ratio=0.0, test_ratio = 0.4)

def calculate_mean_dbh(loader):
    total_dbh = 0.0
    count = 0

    for _, _, y in loader:
        dbh=y['dbh']

        total_dbh += dbh.sum().item()
        count += dbh.numel()

    return total_dbh / count

mean_dbh = calculate_mean_dbh(train_loader)
test_mean_dbh = calculate_mean_dbh(test_loader)
print(f"Train mean training DBH:  {mean_dbh:.4f}")
print(f"Test mean training DBH:  {test_mean_dbh:.4f}")

def evaluate_mean_baseline(train_loader, test_loader):
    mean_dbh = calculate_mean_dbh(train_loader)

    squared_err = 0.0
    absolute_err = 0.0
    count = 0

    for _,_,y in test_loader:
        real = y["dbh"]

        predicted = torch.full_like(real, mean_dbh)

        squared_err += ((predicted - real)**2).sum().item()
        absolute_err += (predicted - real).abs().sum().item()
        count += real.numel()

    mse = squared_err / count
    rmse = mse ** 0.5
    mae = absolute_err / count

    return {
        "mean_dbh": mean_dbh,
        "MAE": mae,
        "RMSE": rmse,
    }

results = evaluate_mean_baseline(train_loader, test_loader)
print(results)
