# -*- coding: utf-8 -*-
# @Time : 2022/5/13 19:28
# @Author : Mengtian Zhang
# @Version : v-dev-0.0
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""

from data.dataset import get_dataset_cifar10, get_class_name_cifar10
import torch
from tools import utils
from tools import tools_for_statistics as tool_stat
from Networks import vgg_exp3
from tqdm import tqdm
from tools.utils_plot import *
import os
from tools.metrics import *
from time import time
from tensorboardX import SummaryWriter

args = utils.get_args()
# args = utils.get_args("--batch-size 128 --gpu 2".split())

utils.seed_everything(args.seed)
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# PATH
output_root_dir = './Outputs/Exp3/'
output_root_dir = os.path.join(output_root_dir, utils.get_timestamp())
output_visual_dir = os.path.join(output_root_dir, 'visualization')
utils.makedirs(output_visual_dir)

utils.save_parameters(output_root_dir, vars(args))

# Dataset & Dataloader
# In Windows system. the num_workers can be set as nothing but 0
# The original code set it as 8, and we changed here temprally
train_set, test_set = get_dataset_cifar10(root='./data')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                           shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                          shuffle=False, num_workers=0)

# Network, loss_fn, optimizer, scheduler
try:
    net_module = getattr(vgg, args.network)
    print(args.network)
except AttributeError:
    net_module = vgg.vgg11
net = net_module(num_classes=10)
net.to(device)
if args.load_checkpoint!=None:
    net.load_state_dict(torch.load(args.load_checkpoint))
    print("load success!")

loss_fn = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
# optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# tensorboard writer for recording temp results
writer = SummaryWriter(os.path.join(output_root_dir, "tensorboard"))


# ************************************* FUNCTIONS **************************************************
def get_confusion_matrix(data_loader: torch.utils.data.DataLoader):
    num_of_class = len(get_class_name_cifar10())
    label_array = list()
    predicted_array = list()
    with torch.no_grad():
        for batch in data_loader:
            data, label = batch
            data = data.to(device)
            label = label.to(device)
            output = net(data)
            _, predicted = torch.max(output.data, 1)

            label_array.append(label)
            predicted_array.append(predicted)
    label_array = torch.cat(label_array).cpu().numpy()
    predicted_array = torch.cat(predicted_array).cpu().numpy()
    confusion_matrix = calculate_confusion_matrix(predicted_array, label_array, num_of_class)
    return confusion_matrix


def run_plot_confusion_matrix(epoch_: int):

    if epoch_ % 10 != 0:
        return

    confusion_matrix_train = get_confusion_matrix(train_loader)
    confusion_matrix_test = get_confusion_matrix(test_loader)
    category_name = get_class_name_cifar10()
    get_heatmap(confusion_matrix_train, row_labels=category_name, col_labels=category_name,
                save_path=os.path.join(output_visual_dir, 'confusion_matrix_train.png'))
    get_heatmap(confusion_matrix_test, row_labels=category_name, col_labels=category_name,
                save_path=os.path.join(output_visual_dir, 'confusion_matrix_test.png'))


def plot_curves(cre: int = -1):
    if cre % 10 != 0:
        return

    # Plot acc and loss curve
    plot_curve_for_train_and_test(None, y_train=result_dict['train_acc_list'], y_test=result_dict['test_acc_list'],
                                  save_path=os.path.join(output_visual_dir, 'acc.png'),
                                  parameter_dict={'title': "Accuracy Curve",
                                                  'xlabel': 'Epoch',
                                                  'ylabel': 'Accuracy'
                                                  })
    plot_curve_for_train_and_test(None, y_train=result_dict['train_loss_list'], y_test=result_dict['test_loss_list'],
                                  save_path=os.path.join(output_visual_dir, 'loss.png'),
                                  parameter_dict={'title': "Loss Curve",
                                                  'xlabel': 'Epoch',
                                                  'ylabel': 'Loss'
                                                  })


def update_result_dict(epoch_: int, train_result_: dict, test_result_: dict):
    result_dict['train_acc_list'].append(train_result_['acc'])
    result_dict['train_loss_list'].append(train_result_['loss'])

    result_dict['test_acc_list'].append(test_result_['acc'])
    result_dict['test_loss_list'].append(test_result_['loss'])

    if test_result_['acc'] > result_dict['test_acc_optim']:
        result_dict['test_acc_optim'] = test_result_['acc']
        result_dict['test_epoch_optim'] = epoch_


def train(epoch_: int):
    net.train()

    loss_stat = tool_stat.ValueStat()
    acc_stat = tool_stat.ValueStat()

    run_bar = tqdm(train_loader, desc=f"[Train] Epoch={epoch_}")
    for batch in run_bar:
        data, label = batch
        data = data.to(device)
        label = label.to(device)

        output = net(data)

        loss = loss_fn(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = get_accuracy(output, label)
        acc_stat.update(acc)

        loss_stat.update(loss.item())

        run_bar.set_postfix({
            "loss": round(loss_stat.get_avg(), 2),
            "acc": round(acc_stat.get_avg(), 4)
        })

    return {
        "loss": loss_stat.get_avg(),
        "acc": acc_stat.get_avg()
    }


def test(epoch_: int):
    net.eval()

    loss_stat = tool_stat.ValueStat()
    acc_stat = tool_stat.ValueStat()

    with torch.no_grad():
        run_bar = tqdm(test_loader, desc=f"[Test] Epoch={epoch_}")
        for batch in run_bar:
            data, label = batch
            data = data.to(device)
            label = label.to(device)

            output = net(data)

            loss = loss_fn(output, label)
            loss_stat.update(loss.item())

            acc = get_accuracy(output, label)
            acc_stat.update(acc)

            run_bar.set_postfix({
                "loss": round(loss_stat.get_avg(), 2),
                "acc": round(acc_stat.get_avg(), 4)
            })

    return {
        "loss": loss_stat.get_avg(),
        "acc": acc_stat.get_avg()
    }


# ************************************* RUNNING ****************************************************

result_dict = {
    'test_acc_optim': 0,
    'test_epoch_optim': -1,
    'train_loss_list': list(),
    'train_acc_list': list(),
    'test_acc_list': list(),
    'test_loss_list': list(),
}

begin_time = time()
for epoch in range(args.epochs + 1):
    train_result = train(epoch)
    test_result = test(epoch)
    scheduler.step()

    writer.add_scalar("train_acc", train_result['acc'], epoch)
    writer.add_scalar("train_loss", train_result['loss'], epoch)
    writer.add_scalar("test_acc", test_result['acc'], epoch)
    writer.add_scalar("test_loss", test_result['loss'], epoch)

    update_result_dict(epoch, train_result, test_result)
    plot_curves(epoch)
    run_plot_confusion_matrix(epoch)
writer.close()
end_time = time()
run_time = int(end_time - begin_time)
torch.save(net.state_dict().copy(),os.path.join(output_root_dir,"net.pth"))

# *********************************** RESULT-OUTPUT *************************************************
output_str = ""
output_str += "\n\n" + '*' * 50 + '\n'
output_str += '\t*' + "Optimal epoch: {}".format(result_dict['test_epoch_optim']) + '\n'
output_str += '\t*' + "Optimal test accuracy: {:.2f}".format(result_dict['test_acc_optim']) + '\n'
output_str += '\t*' + "Total running time: {}h - {}m - {}s".format(run_time//3600, run_time//60%60, run_time%3600%60) + '\n'
output_str += '*' * 50 + '\n'

print(output_str)

with open(os.path.join(output_root_dir, 'result.txt'), 'w') as file:
    file.write(output_str)
