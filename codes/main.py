from torch.optim import Adam
from tqdm import tqdm
from utils import *
from train_dblp import train_dblp


def train(model, data, device, epochs):
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    min_epochs = 5
    best_val_acc = 0
    final_best_acc = 0
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data['author'].y[data.train_mask])
        loss.backward()
        optimizer.step()

        # validation
        val_acc, val_loss = evaluate(model, data, data.val_mask, device)
        test_acc, test_loss = evaluate(model, data, data.test_mask, device)
        if epoch + 1 > min_epochs and val_acc > best_val_acc:
            best_val_acc = val_acc
            final_best_acc = test_acc
        print('train_loss {:.5f} val_acc {:.3f} test_acc {:.3f}'
              .format(loss.item(), val_acc, test_acc))

    print(f'best acc: {final_best_acc}')
    return final_best_acc


def evaluate(model, data, mask, device):
    model.eval()
    with torch.no_grad():
        out = model(data)
        loss_function = torch.nn.CrossEntropyLoss().to(device)
        loss = loss_function(out[mask], data['author'].y[mask])
    _, pred = out.max(dim=1)
    correct = int(pred[mask].eq(data['author'].y[mask]).sum().item())
    acc = correct / int(mask.sum())

    return acc, loss.item()


def cal_mean_std(array_data):
    mean = np.mean(array_data)
    std = np.std(array_data)
    return mean, std


def generate_random_seeds(n, seed_range=(0, 10000)):
    seeds = np.random.randint(seed_range[0], seed_range[1], size=n)
    return seeds


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_num = 3
    accuracy_array = []
    seeds = generate_random_seeds(train_num)
    for seed in seeds:
        set_seed(seed)

        best = train_dblp(device)
        # train_imdb(device)
        # train_acm(device)
        # train_yelp(device)
        accuracy_array.append(best["acc"])

    mean, std = cal_mean_std(accuracy_array)
    print(f"mean:{mean:.4f}, std:{std:.4f}")

