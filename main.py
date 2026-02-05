import pickle

import dgl
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from utils_cluster import *
from train_dblp import train_dblp
from train_imdb import train_imdb
from train_acm import train_acm
from train_yelp import train_yelp
from train_aminer import train_aminer
from train_freebase import train_freebase
from picture import *
import os


def train(model, data, device, epochs):
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
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

    directory = os.path.join(os.getcwd(), "pkl")
    if not os.path.exists(directory):
        os.makedirs(directory)

    database = 'dblp'  # yelp, dblp, imdb, acm, aminer, freebase
    train_num = 1
    accuracy_array = []
    micro_array = []
    macro_array = []
    seeds = generate_random_seeds(train_num)
    for seed in seeds:
        set_seed(seed)

        train_func = 'train_' + database + '(device)'
        best = eval(train_func)

        accuracy_array.append(best["acc"])
        micro_array.append(best["micro_f1"])
        macro_array.append(best["macro_f1"])

    with open(os.path.join(directory, database + '.pkl'), 'wb') as pickle_file:
        pickle.dump({"acc": accuracy_array, "micro_f1": micro_array, "macro_f1": macro_array}, pickle_file)

    mean, std = cal_mean_std(accuracy_array)
    print(f"acc mean:{mean:.4f}, std:{std:.4f}")

    mean, std = cal_mean_std(micro_array)
    print(f"micro_f1 mean:{mean:.4f}, std:{std:.4f}")

    mean, std = cal_mean_std(macro_array)
    print(f"macro_f1 mean:{mean:.4f}, std:{std:.4f}")
