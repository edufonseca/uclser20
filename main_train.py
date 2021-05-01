import pprint
import sys
from subprocess import Popen, PIPE
# importing numpy to fix https://github.com/pytorch/pytorch/issues/37377
import numpy
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torchvision import transforms
import torch.optim as optim
from FSDnoisy18k.dataset.FSDnoisy18k import *

sys.path.append('./src')
sys.path.append('./da')
from data_augment import SpecAugment, RandomGaussianBlur, GaussNoise, RandTimeShift, RandFreqShift, TimeReversal, Compander
from rnd_resized_crop import RandomResizedCrop_diy

from utils_train_eval import *

import models.audio_models as audio_mod
import models.resnetAudio as resnet_mod
import models.MLP_head as mlp
import yaml
import random

import os
import argparse

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def parse_args():
    parser = argparse.ArgumentParser(description='Code for MTG-DCU collaboration')
    parser.add_argument('-p', '--params_yaml', dest='params_yaml', action='store', required=False, type=str)
    config = parser.parse_args()
    print('\nYaml file with parameters defining the experiment: %s\n' % str(config.params_yaml))
   
    # Read parameters file from yaml passed by argument
    args = yaml.load(open(config.params_yaml))

    return args

def my_collate(batch):
    # for validation and test
    imgs, targets, index = zip(*batch)
    individual_sizes = []
    for i in range(len(imgs)):
        individual_sizes.append(imgs[i].size()[0])
    #Repeat tensors to know each sample index!!

    return torch.cat(imgs), torch.from_numpy(np.concatenate((targets))), torch.from_numpy(np.repeat(index, individual_sizes))

def data_config(args):

    output, _ = Popen('uname', stdout=PIPE).communicate()
    print(output.decode("utf-8"))

    trainset, valset, testset = get_dataset(args)

    mean = [trainset.data_mean]
    std = [trainset.data_std]

    train_transform = transforms.Compose([
        RandTimeShift(do_rand_time_shift=args['da']['do_rand_time_shift'], Tshift=args['da']['Tshift']),
        RandFreqShift(do_rand_freq_shift=args['da']['do_rand_freq_shift'], Fshift=args['da']['Fshift']),
        RandomResizedCrop_diy(do_randcrop=args['da']['do_randcrop'], scale=args['da']['rc_scale'],
                              ratio=args['da']['rc_ratio']),
        transforms.RandomApply([TimeReversal(do_time_reversal=args['da']['do_time_reversal'])], p=0.5),
        Compander(do_compansion=args['da']['do_compansion'], comp_alpha=args['da']['comp_alpha']),
        SpecAugment(do_time_warp=args['da']['do_time_warp'], W=args['da']['W'],
                    do_freq_mask=args['da']['do_freq_mask'], F=args['da']['F'], m_f=args['da']['m_f'],
                    reduce_mask_range=args['da']['reduce_mask_range'],
                    do_time_mask=args['da']['do_time_mask'], T=args['da']['T'], m_t=args['da']['m_t'],
                    mask_val=args['da']['mask_val']),
        GaussNoise(stdev_gen=args['da']['awgn_stdev_gen']),
        RandomGaussianBlur(do_blur=args['da']['do_blur'], max_ksize=args['da']['blur_max_ksize'],
                           stdev_x=args['da']['blur_stdev_x']),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),

    ])

    trainset.transform = train_transform
    valset.transform = test_transform
    testset.transform = test_transform

    trainset.pslab_transform = test_transform


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args['learn']['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    track_loader = torch.utils.data.DataLoader(trainset, batch_size=args['learn']['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args['learn']['test_batch_size'], collate_fn=my_collate, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args['learn']['test_batch_size'], collate_fn=my_collate, shuffle=False, num_workers=8, pin_memory=True)

    print('############# Data loaded #############')

    return train_loader, val_loader, test_loader, track_loader


def main(args):

    print('\nExperimental setup:')
    print('pctrl=')
    pprint.pprint(args['ctrl'], width=1, indent=4)
    print('plearn=')
    pprint.pprint(args['learn'], width=1, indent=4)
    print('pda=')
    pprint.pprint(args['da'], width=1, indent=4)

    job_start = time.time()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['learn']['cuda_dev'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args['learn']['seed_initialization'])  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args['learn']['seed_initialization'])  # GPU seed

    random.seed(args['learn']['seed_initialization'])  # python seed for image transformation

    train_loader, val_loader, test_loader, track_loader = data_config(args)
    st = time.time()

    # select model===========================================================================
    # select model===========================================================================
    # note model outputs a certain embedding size (h_i, h_j) which must match input to head
    if args['learn']['network'] == 'res18':
        model = resnet_mod.resnet18(args, num_classes=args["learn"]["num_classes"]).to(device)

        if args['learn']['downstream']==1:
            model.load_state_dict(torch.load("pth/ResNet_best.pth"))
            if args['learn']['lin_eval']==1:
                # freeze weights for linear evaluation
                for param in model.parameters():
                    param.requires_grad = False
            model.fc = torch.nn.Linear(512, args["learn"]["num_classes"]).to(device)

        if args['learn']['method'] == 'Contrastive':
            model_head = mlp.MLPHead(args, in_channels=512, mlp_hidden_size=512).to(device)

    elif args['learn']['network'] == 'vgg_relu':
        model = audio_mod.VGGlike_small_emb_relu(args, num_classes=args["learn"]["num_classes"]).to(device)

        if args['learn']['downstream']==1:
            model.load_state_dict(torch.load("pth/VGG_best.pth"))
            if args['learn']['lin_eval'] == 1:
                # freeze weights for linear evaluation
                for param in model.parameters():
                    param.requires_grad = False
            model.dense_out = torch.nn.Linear(512, args["learn"]["num_classes"]).to(device)

        if args['learn']['method'] == 'Contrastive':
            model_head = mlp.MLPHead(args, in_channels=args["learn"]["embed_size"], mlp_hidden_size=512).to(device)

    elif args['learn']['network'] == 'crnn':
        model = audio_mod.CRNN_tagger(args, num_classes=args["learn"]["num_classes"]).to(device)

        if args['learn']['downstream'] == 1:
            model.load_state_dict(torch.load("pth/CRNN_best.pth"))
            if args['learn']['lin_eval'] == 1:
                # freeze weights for linear evaluation
                for param in model.parameters():
                    param.requires_grad = False
            model.dense = torch.nn.Linear(64, args["learn"]["num_classes"]).to(device)

        if args['learn']['method'] == 'Contrastive':
            model_head = mlp.MLPHead(args, in_channels=64, mlp_hidden_size=512).to(device)

    print('Total params: {:.2f} M'.format((sum(p.numel() for p in model.parameters()) / 1000000.0)))



    # opt, lr scheduling, side results===========================================================================
    # opt, lr scheduling, side results===========================================================================
    milestones = args['learn']['M']

    if args['learn']['method'] == "Contrastive":
        if isinstance(args['learn']['network'], str):
            if args['learn']['opt'] == 'sgd':
                optimizer = optim.SGD(list(model.parameters()) + list(model_head.parameters()), lr=args['learn']['lr'], momentum=args['learn']['momentum'], weight_decay=float(args['learn']['wd']))
            elif args['learn']['opt'] == 'adam':
                optimizer = optim.Adam(list(model.parameters()) + list(model_head.parameters()), lr=args['learn']['lr'], weight_decay=float(args['learn']['wd']))
    else:
        optimizer = optim.SGD(model.parameters(), lr=args['learn']['lr'], momentum=args['learn']['momentum'], weight_decay=float(args['learn']['wd']))


    if args["learn"]["lr_schedule"] == 'multistep':
        # by default we use MultiStepLR. If milestones is [], we have a constant lr
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    elif args["learn"]["lr_schedule"] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=100, threshold=0.0001,
                                                         threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08,
                                                         verbose=True)

    ##Lists to store accuracies and losses
    loss_train_epoch, loss_val_epoch, loss_test_epoch = [], [], []
    acc_train_per_epoch, acc_val_per_epoch, acc_val_balanced_per_epoch, acc_val_balanced_top3_per_epoch, \
    acc_test_per_epoch, acc_test_balanced_per_epoch, acc_test_balanced_top3_per_epoch = [], [], [], [], [], [], []

    ##Paths to save results and models
    exp_path = os.path.join('./results_models/', 'models_' + args['learn']['network'] + '_{0}_SI{1}'.format(args['learn']['experiment_name'],
                                                                                             args['learn']['seed_initialization']))
    res_path = os.path.join('./results_models/', 'metrics_' + args['learn']['network'] + '_{0}_SI{1}'.format(args['learn']['experiment_name'],
                                                                                       args['learn']['seed_initialization']))

    if not os.path.isdir(res_path):
        os.makedirs(res_path)
    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)

    cont = 0

    # training loop===========================================================================
    # training loop===========================================================================
    for epoch in range(args['learn']['initial_epoch'], args['learn']['epoch'] + 1):
        print('######## Epoch: {}'.format(epoch))

        # train for one epoch
        if args['learn']['method'] == "CE": # Train with cross-entropy
            loss_train, top5_train, top1_train, train_time = train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, args['learn']['num_classes'])

        elif args['learn']['method'] == "Contrastive": # Train with unsupervised contrastive learning
            loss_train, top5_train, top1_train, train_time = train_Contrastive(args, model, model_head, device, train_loader, optimizer, epoch, args['learn']['num_classes'])

        if args["learn"]["lr_schedule"] == 'multistep':
            scheduler.step()

        print('######## Validation ########')
        if args["learn"]["method"] == "Contrastive":
            acc_val, acc_val_top3, acc_val_balanced, acc_val_balanced_top3, acc_val_per_class = eval_model_contrastive(args, model, model_head, device, val_loader, 200)
            loss_val = 0.0
            print('\nValidation set. Loss: {:.4f}, Accuracy: {:.4f}\n'.format(loss_val, acc_val_balanced))
        else:
            loss_val, acc_val, acc_val_balanced, acc_val_per_class = eval_model(args, model, device, val_loader)
            print('\nValidation set. Loss: {:.4f}, Accuracy: {:.4f}\n'.format(loss_val, acc_val_balanced))
            acc_val_top3 = 0.0
            acc_val_balanced_top3 = 0.0

        if args["learn"]["lr_schedule"] == 'plateau':
            scheduler.step(acc_val_balanced)

        # test
        print('######## Test ########')
        # loss_test, acc_test, acc_test_balanced, acc_test_per_class = 0.0, 0.0, 0.0, 0.0
        if args["learn"]["method"] == "Contrastive":
            acc_test, acc_test_top3, acc_test_balanced, acc_test_balanced_top3, acc_test_per_class = eval_model_contrastive(args, model, model_head, device, test_loader, 200)
            loss_test = 0.0
            print('\nTest set. Loss: {:.4f}, Accuracy: {:.4f}\n'.format(loss_test, acc_test_balanced))
        else:
            loss_test, acc_test, acc_test_balanced, acc_test_per_class = eval_model(args, model, device, test_loader)
            print('\nTest set. Loss: {:.4f}, Accuracy: {:.4f}\n'.format(loss_test, acc_test_balanced))
            acc_test_top3 = 0.0
            acc_test_balanced_top3 = 0.0


        loss_train_epoch += [loss_train]
        loss_val_epoch += [loss_val]
        loss_test_epoch += [loss_test]

        acc_train_per_epoch += [top1_train]
        acc_val_per_epoch += [acc_val]
        acc_val_balanced_per_epoch += [acc_val_balanced]
        acc_val_balanced_top3_per_epoch += [acc_val_balanced_top3]
        acc_test_per_epoch += [acc_test]

        acc_test_balanced_per_epoch += [acc_test_balanced]
        acc_test_balanced_top3_per_epoch += [acc_test_balanced_top3]

        if len(acc_val_balanced_per_epoch)>=50:
            acc_avg_val = np.asarray(acc_val_balanced_per_epoch[-50:]).mean()
            acc_avg_test = np.asarray(acc_test_balanced_per_epoch[-50:]).mean()
        else:
            acc_avg_val = 0.0
            acc_avg_test = 0.0


        print('Epoch time: {:.2f} seconds\n'.format(time.time()-st))
        st = time.time()

        if epoch == args['learn']['initial_epoch']:
            best_acc_val = acc_val_balanced
            best_acc_val_top3 = acc_val_balanced_top3
            acc_test_for_best_acc_val = acc_test_balanced
            acc_test_for_best_acc_val_top3 = acc_test_balanced_top3
            best_acc_val_pc = acc_val_per_class
            acc_test_for_best_acc_val_pc = acc_test_per_class

            counter_pat_es = 0   # init counter counter for ES

            snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_bestAccVal_%.5f' % (
                epoch, loss_val, acc_val_balanced, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
        else:

            if acc_val_balanced > best_acc_val:
                best_acc_val = acc_val_balanced
                best_acc_val_top3 = acc_val_balanced_top3
                acc_test_for_best_acc_val = acc_test_balanced
                acc_test_for_best_acc_val_top3 = acc_test_balanced_top3
                best_acc_val_pc = acc_val_per_class
                acc_test_for_best_acc_val_pc = acc_test_per_class

                counter_pat_es = 0   # reset ES patience counter

                if cont > 0:
                    try:
                        os.remove(os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
                        os.remove(os.path.join(exp_path, snapBest + '.pth'))
                    except OSError:
                        pass
                snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_bestAccVal_%.5f' % (
                    epoch, loss_val, acc_val_balanced, best_acc_val)
                torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
                torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))

            else:
                # no acc_val_balanced improvement, increment counter for ES
                counter_pat_es += 1

        cont += 1

        if epoch == args['learn']['epoch']:
            snapLast = 'last_epoch_ckpt'
            torch.save(model.state_dict(), os.path.join(exp_path, snapLast + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapLast + '.pth'))


        # Save losses:
        np.save(res_path + '/' + 'loss_epoch_train.npy', np.asarray(loss_train_epoch))
        np.save(res_path + '/' + 'loss_epoch_val.npy', np.asarray(loss_val_epoch))
        np.save(res_path + '/' + 'loss_epoch_test.npy', np.asarray(loss_test_epoch))

        # save accuracies:
        np.save(res_path + '/' + 'accuracy_per_epoch_train.npy', np.asarray(acc_train_per_epoch))
        np.save(res_path + '/' + 'accuracy_per_epoch_val.npy', np.asarray(acc_val_per_epoch))
        np.save(res_path + '/' + 'accuracy_per_epoch_val_balanced.npy', np.asarray(acc_val_balanced_per_epoch))
        np.save(res_path + '/' + 'accuracy_per_epoch_val_balanced_top3.npy', np.asarray(acc_val_balanced_top3_per_epoch))
        np.save(res_path + '/' + 'accuracy_per_epoch_test.npy', np.asarray(acc_test_per_epoch))
        np.save(res_path + '/' + 'accuracy_per_epoch_test_balanced.npy', np.asarray(acc_test_balanced_per_epoch))
        np.save(res_path + '/' + 'accuracy_per_epoch_test_balanced_top3.npy', np.asarray(acc_test_balanced_top3_per_epoch))

        # early stopping
        if counter_pat_es == args['learn']['early_stopping_patience'] and args['learn']['early_stopping_do']:
            print('====Training HALTED as run out of patience of {} epochs.'.format(args['learn']['early_stopping_patience']))
            break


    print('Best Validation Accuracy: {:.3f}'.format(best_acc_val))
    print('Test Accuracy for Best Val Accuracy: {:.3f}\n'.format(acc_test_for_best_acc_val))
    print('Last_Validation_Accuracy: {:.3f}'.format(acc_val_balanced))
    print('Last_Test_Accuracy: {:.3f}\n'.format(acc_test_balanced))
    print('Average Last Validation Accuracy: {:.3f}'.format(acc_avg_val))
    print('Average Last Test Accuracy: {:.3f}\n'.format(acc_avg_test))
    print('Best Top-3 Validation Accuracy: {:.3f}'.format(best_acc_val_top3))
    print('Test Top-3 Accuracy for Best Val Accuracy: {:.3f}\n'.format(acc_test_for_best_acc_val_top3))


    print('PER CLASS Val Accuracy (best):')
    for i in range(len(best_acc_val_pc)):
        print('{}: {:.3f}'.format('class_name', best_acc_val_pc[i]))

    print('\nPER CLASS Test Accuracy (best):')
    for i in range(len(acc_test_for_best_acc_val_pc)):
        print('{}: {:.3f}'.format('class_name', acc_test_for_best_acc_val_pc[i]))


    print('\n=============================Job finalized==========================================================')
    print('Time elapsed for one job: {:.1f} min'.format((time.time()-job_start)/60.0))
    print('====================================================================================================\n')

if __name__ == "__main__":
    args = parse_args()

    # train
    main(args)
