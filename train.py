# coding:utf-8
import os
import time
import torch
import random
import numpy as np
import argparse

from torch.utils import data
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn import functional as F
from torch.backends import cudnn
import torchvision.models as models

from ptclassifaction.metrics import averageMeter, runningScore
from ptclassifaction.utils import make_dir, get_logger
from ptclassifaction.loader import get_loader
from ptclassifaction.models import get_model

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
devices_ids = 0

def get_arguments():
    parser = argparse.ArgumentParser(description="Pytorch Classification Master")
    parser.add_argument("--config-name", type=str, default='resnet_classifaction',help="")
    parser.add_argument("--cuda_devices", type=str, default='0', help="")
    parser.add_argument("--model-arch", type=str, default='resnet101', help="")
    parser.add_argument("--dataset", type=str, default='clasegloader',help="")
    parser.add_argument("--train-split", type=str, default='train', help="")
    parser.add_argument("--test-split", type=str, default='test', help="")
    parser.add_argument("--n-classes", type=int, default=3, help="")
    parser.add_argument("--img-rows", type=int, default=256, help="")
    parser.add_argument("--img-cols", type=int, default=256, help="")
    parser.add_argument("--input-channels", type=int, default=3, help="")
    parser.add_argument("--data-path", type=str, default='../../Data/voc_aug_cnv3/', help="")
    parser.add_argument("--data-name", type=str, default='voc_aug_cnv3', help="")
    parser.add_argument("--fold-series", type=str, default='1', help="")
    parser.add_argument("--seed", type=int, default=1334, help="")
    parser.add_argument("--train-iters", type=int, default=200, help="")
    parser.add_argument("--batch-size", type=int, default=8, help="")
    parser.add_argument("--val-interval", type=int, default=10, help="")
    parser.add_argument("--n-workers", type=int, default=16, help="")
    parser.add_argument("--print-interval", type=int, default=1, help="")
    parser.add_argument("--optimizer-name", type=str, default='amad', help="")
    parser.add_argument("--lr", type=float, default=1.0e-5, help="")
    parser.add_argument("--weight-decay", type=float, default=0.08, help="")
    parser.add_argument("--momentum", type=float, default=0.99, help="")
    parser.add_argument("--loss-name", type=str, default='cam_loss', help="")
    parser.add_argument("--pkl-path", type=str, default='./pkls', help="")
    parser.add_argument("--resume", type=str, default='', help="")

    return parser.parse_args()


def train(args, writer, logger):

    # Setup seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    train_loader = get_loader(args.dataset)(
        args.data_path,
        is_transform=True,
        split=args.train_split,
        img_size=(args.img_rows, args.img_cols),
        n_classes=args.n_classes,
        fold_series=args.fold_series,
    )

    test_loader = get_loader(args.dataset)(
        args.data_path,
        is_transform=True,
        split=args.test_split,
        img_size=(args.img_rows, args.img_cols),
        n_classes=args.n_classes,
        fold_series=args.fold_series,
    )

    trainloader = data.DataLoader(
        train_loader,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        shuffle=True,
    )

    testloader = data.DataLoader(
        test_loader,
        batch_size=args.batch_size,
        num_workers=0
    )
    # Setup Metrics
    running_metrics_val = runningScore(args.n_classes)

    # Setup Model
    model = get_model(args.model_arch, pretrained=True, num_classes=args.n_classes,
                      input_channels=args.input_channels).to(device)
    # summary(model, (cfg["data"]['input_channels'],\
    #                              cfg["data"]["img_rows"], cfg["data"]["img_cols"]))
    # model = models.resnet101(pretrained=True)
    # fc_features = model.fc.in_features
    # model.fc = torch.nn.Linear(fc_features, n_classes)

    model = model.cuda(device=devices_ids)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.08)
    logger.info("Using optimizer {}".format(optimizer))

    # apex
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    loss_fn = torch.nn.CrossEntropyLoss()
    from ptclassifaction.loss.cam_loss import CAMLoss
    loss_fn2 = CAMLoss()
    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    # reload from checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(args.resume)
            )
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(args.resume))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_acc = -100
    epoch_iter = start_iter
    running_loss = 0.0
    for epoch in range(epoch_iter, args.train_iters):
        # train_inter
        loss = 0.0
        for (train_images, train_seg_labels, train_cla_labels) in trainloader:
            start_ts = time.time()

            model.train()
            train_images = train_images.to(device)
            train_cla_labels = train_cla_labels.to(device)
            train_seg_labels = train_seg_labels.to(device)

            optimizer.zero_grad()
            outputs = model(train_images)

            ###################################################################################################
            features_blo = []

            def hook_feature(module, input, outputs):
                features_blo.append(outputs)

            handle = model._modules.get('layer4').register_forward_hook(hook_feature)

            # get the softmax weight
            params = list(model.parameters())
            weight_softmax = params[-2].data.squeeze()

            logit = model(train_images)

            h_x = F.softmax(logit, dim=1).data.squeeze()
            _, idx = h_x.sort(1)

            handle.remove()
            features_blobs = features_blo[0]
            ###################################################################################################

            loss = loss_fn(outputs, train_cla_labels)
            # loss = loss_fn2(outputs, train_cla_labels, train_seg_labels, features_blobs, weight_softmax, idx)

            loss.backward()
            # apex
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()

            running_loss += loss.item()

            time_meter.update(time.time() - start_ts)

        # print_interval
        if (epoch + 1) % args.print_interval == 0:
            fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}".format(
                epoch + 1,
                args.train_iters,
                running_loss / args.print_interval,
                time_meter.avg / args.batch_size,
            )
            running_loss = 0.0
            print(fmt_str)
            # logger.info(print_str)
            writer.add_scalar("loss/train_loss", loss.item(), epoch + 1)
            # histograms and multi-quantile line graphs
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch + 1)
            time_meter.reset()

        # val_interval
        if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.train_iters:
            model.eval()
            with torch.no_grad():
                for (val_images, val_seg_labels, val_labels) in testloader:
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)

                    outputs = model(val_images)
                    val_loss = loss_fn(input=outputs, target=val_labels)

                    _, predicted = torch.max(outputs.data, 1)

                    running_metrics_val.update(predicted.cuda().data.cpu().numpy(),\
                                               val_labels.cuda().data.cpu().numpy())
                    val_loss_meter.update(val_loss.item())

            writer.add_scalar("loss/val_loss", val_loss_meter.avg, epoch + 1)
            logger.info("Val Iter %d Loss: %.4f" % (epoch + 1, val_loss_meter.avg))

            score = running_metrics_val.get_scores()
            for k, v in score.items():
                print(k, v)
                logger.info("{}: {}".format(k, v))
                writer.add_scalar("val_metrics/{}".format(k), v, epoch + 1)

            val_loss_meter.reset()
            running_metrics_val.reset()

            if score["Overall Acc: \t"] >= best_acc:
                best_acc = score["Overall Acc: \t"]
                state = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    # "scheduler_state": scheduler.state_dict(),
                    "best_acc": best_acc,
                }

                save_path = os.path.join(
                    args.pkl_path,
                    os.path.basename(args.config_name)[:-4],
                    str(run_id)
                )
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                torch.save(state, save_path + "/{}_{}_best_model.pkl".format(
                    args.model_arch, args.data_name))

        writer.close()


if __name__ == "__main__":
    args = get_arguments()

    make_dir("./runs")
    make_dir(args.pkl_path)

    run_dir = os.path.join("runs", os.path.basename(args.config_name))
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    run_list = os.listdir(run_dir)
    run_id_list = [run_list[-4:] for run_list in run_list]
    if run_id_list:
        run_id = args.data_name + '_' + str(int(max(run_id_list)) + 1)
        os.makedirs(os.path.join(run_dir, run_id))

    else:
        run_id = args.data_name + '_' + str(1001)
        os.makedirs(os.path.join(run_dir, run_id))

    print("RUNDIR: {}".format(os.path.join(run_dir, run_id)))
    logdir = os.path.join(run_dir, run_id)
    writer = SummaryWriter(log_dir=logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    train(args, writer, logger)