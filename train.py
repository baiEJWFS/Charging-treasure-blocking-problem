import argparse
import time
import warnings

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

from data import *
from layers.modules import MultiBoxLoss
from ssd import build_ssd
from utils.augmentations import SSDAugmentation

warnings.filterwarnings('ignore')


# device = torch.device('cuda:1')


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default=dataset,
                    type=str, help='SIXRAY')
parser.add_argument('--dataset_root', default=SIXray_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default=basenet,
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=batch_size, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=resume, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=start_iter, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=num_workers, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=is_cuda, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=lr, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=momentum, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=weight_decay, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=gamma, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=visdom, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default=train_save_folder,
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'SIXRAY':
        if args.dataset_root == SIXray_ROOT:
            if not os.path.exists(SIXray_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default SIXRAY dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = SIXray_ROOT
        cfg = sixray
        # list=[]
        # print("cfg", cfg)
        dataset = SIXrayDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        # list.append(dataset)
        # print("dataset:", list)
    else:
        print("ERRO: Only Using default SIXRAY dataset_root.")
        return

    if args.visdom:
        import visdom
        global viz
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.cuda:
        # device_ids = [0,1]
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()
        # net = net.to(device)

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 1
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('epoch_size:', epoch_size)
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True, drop_last=True)
    # data_loader = data.DataLoader(dataset, args.batch_size,
    #                               num_workers=args.num_workers,
    #                               shuffle=True, drop_last=True)
    print(len(data_loader))
    print('load data over ~')
    # create batch iterator
    # batch_iterator = iter(data_loader)+

    iter = 0
    for epoch in range(1, 100):
        for i, data_ in enumerate(data_loader):
            iter += 1
            images, targets = data_
            print(targets)
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
                # images = Variable(images.to(device))
                # targets = [Variable(ann.to(device), volatile=True) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]
            # forward
            t0 = time.time()

            out = net(images)
            # print('------output',(out[0].size(),out[1].size()))
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.data
            conf_loss += loss_c.data

            if iter % 1 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iter) + ' || Loss: %.4f ||' % (loss.data), end=' ')

            if args.visdom:
                update_vis_plot(iter, loss_l.data, loss_c.data,
                                iter_plot, epoch_plot, 'append')

        if epoch % 1 == 0:
            print('Saving state, iter:', epoch)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_SIXRAY_' +
                       repr(epoch) + '.pth')
        if args.visdom and epoch != 0:
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0

    # for iteration in range(args.start_iter, cfg['max_iter']):
    #     if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
    #         update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
    #                         'append', epoch_size)
    #         # reset epoch loss counters
    #         loc_loss = 0
    #         conf_loss = 0
    #         epoch += 1
    #
    #     if iteration in cfg['lr_steps']:
    #         step_index += 1
    #         adjust_learning_rate(optimizer, args.gamma, step_index)
    #
    #     # load train data
    #     try:
    #         images, targets = next(batch_iterator)
    #         print(images,targets)
    #         print('22222iter:', iter)
    #     except StopIteration:
    #         print('111111iter:',iter)
    #
    #     # print('Go On ')
    #     # print('------input',targets)
    #     if args.cuda:
    #         images = Variable(images.cuda())
    #         targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
    #         # images = Variable(images.to(device))
    #         # targets = [Variable(ann.to(device), volatile=True) for ann in targets]
    #     else:
    #         images = Variable(images)
    #         targets = [Variable(ann, volatile=True) for ann in targets]
    #     # forward
    #     t0 = time.time()
    #
    #     out = net(images)
    #     # print('------output',(out[0].size(),out[1].size()))
    #     # backprop
    #     optimizer.zero_grad()
    #     loss_l, loss_c = criterion(out, targets)
    #     loss = loss_l + loss_c
    #     loss.backward()
    #     optimizer.step()
    #     t1 = time.time()
    #     loc_loss += loss_l.data
    #     conf_loss += loss_c.data
    #
    #     if iteration % 10 == 0:
    #         print('timer: %.4f sec.' % (t1 - t0))
    #         print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data), end=' ')
    #
    #     if args.visdom:
    #         update_vis_plot(iteration, loss_l.data, loss_c.data,
    #                         iter_plot, epoch_plot, 'append')
    #
    #     if iteration != 0 and iteration % 100 == 0:
    #         print('Saving state, iter:', iteration)
    #         torch.save(ssd_net.state_dict(), 'weights/ssd300_SIXRAY_' +
    #                    repr(iteration) + '.pth')
    # torch.save(ssd_net.state_dict(),
    #            args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    print('调整学习率：', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        print('初始化', iteration)
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
