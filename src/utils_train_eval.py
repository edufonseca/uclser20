from __future__ import print_function
import torch.nn as nn
import numpy as np
from utils.AverageMeter import AverageMeter
from utils.criterion import *
import time
import warnings
warnings.filterwarnings('ignore')

global_step = 0

def train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, num_classes):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    criterion = nn.CrossEntropyLoss()
    counter = 1

    for examples, labels, index in train_loader:

        examples, labels, index = examples.to(device), labels.to(device), index.to(device)

        # compute output & loss
        outputs = model(examples)
        loss = criterion(outputs, labels)

        # compute accuracy & updates
        prec1, prec5 = accuracy_v2(outputs, labels, top=[1, 5])
        train_loss.update(loss.item(), examples.size(0))
        top1.update(prec1.item(), examples.size(0))
        top5.update(prec5.item(), examples.size(0))

        # compute gradient and do optimizer step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, counter * len(examples), len(train_loader.dataset),
                       100. * counter / len(train_loader), loss.item(),
                prec1,
                optimizer.param_groups[0]['lr']))
        counter += 1

    return train_loss.avg, top5.avg, top1.avg, batch_time.sum


def train_Contrastive(args, model, model_head, device, train_loader, optimizer, epoch, num_classes):
    """
    This train step is based on one single encoder shared across both simCLR branches.

    :param args:
    :param model:
    :param model_head:
    :param device:
    :param train_loader:
    :param optimizer:
    :param epoch:
    :param num_classes:
    :param sample_loss_per_epoch_tensor:
    :return:
    """

    batch_time = AverageMeter()
    train_loss = AverageMeter()

    # switch to train mode
    model.train()
    model_head.train()
    end = time.time()

    for ex1, ex2, labels, index in train_loader:

        ex1, ex2, labels, index = ex1.to(device), ex2.to(device), labels.to(device), index.to(device)
        # note ex1 and ex2 are batches of examples
        bsz = ex1.shape[0]

        # get the encoders' output (h_i, h_j), which feed the projection head
        if args['learn']['pretrained']:
            # model downloaded is less flexible
            embedi = model(ex1)
            embedj = model(ex2)
        else:
            # typical case: we extract embeddings before the fc layer
            _, embedi = model(ex1)
            _, embedj = model(ex2)

        # get metric embeddings that feed the loss (z_i, z_j)
        z_i = model_head(embedi)
        z_j = model_head(embedj)

        # concatenate the two batches: from N to 2N
        zs = torch.cat([z_i, z_j], dim=0)

        # Compute logits (pairwise comparison + temperature normalization)
        # embeddings are L2 normalized within the MLP, hence this dot product yields cosine similarity
        pairwise_comp = torch.div(torch.matmul(zs, zs.t()), args["learn"]["temp"])
        logits = pairwise_comp

        # Unsupervised: one only positive by augmentation
        mask = torch.eye(bsz, dtype=torch.float32).to(device)

        # Positives mask: 3 parallel diagonals marking the same patches AND positives by augment
        mask = mask.repeat(2, 2)

        # Mask-out self-contrast cases
        # logits mask (all elements are one, except main diagonal with zeros, i.e. self-contrast)
        logits_mask = (torch.ones_like(mask) - torch.eye(2 * bsz).to(device))
        # Final positives mask after zeroing the diagonal, i.e. self-contrast cases
        mask = mask * logits_mask
        # only left are the positives by augmentation (one per example): 2 parallel non-main diagonals

        # Compute log_prob for 2N-1 pairs. This is the denominator
        # in SimCLR: all negs (2N-2) and the only pos by augmentation
        exp_logits_den = torch.exp(logits) * logits_mask  # remove diagonal (self-contrast)
        exp_logits_den = torch.log(exp_logits_den.sum(1, keepdim=True) + 1e-10)
        # we sum each slice of exp_logits_den: sim of all the negs in the denominator for a given pair of positives (i,j)

        # Numerator: compute log_prob for positives (not yet masked)
        exp_logits_pos = torch.log(torch.exp(logits) + 1e-10)

        # log_prob is a subtraction after log(.) (for all samples, needs masking)
        log_prob = exp_logits_pos - exp_logits_den

        # Compute mean of log-likelihood over positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos
        loss = loss.view(2, bsz)
        loss = loss.mean()

        train_loss.update(loss.item(), 2*bsz)

        # compute gradient and do opt step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return train_loss.avg, 0.0, 0.0, batch_time.sum


def eval_model(args, model, device, test_loader):
    """
    Evaluation function for supervised learning. Metrics are balanced (macro) and unbalanced (micro) top-1 accuracies.
    :param args:
    :param model:
    :param device:
    :param test_loader:
    :return:
    """

    model.eval()
    test_loss = AverageMeter()
    top1 = AverageMeter()

    # init vars for per class metrics
    correct = np.zeros((args['learn']['num_classes'],))
    total = np.zeros((args['learn']['num_classes'],))
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (examples, labels, index) in enumerate(test_loader):
            # index contains the clip ID, i.e. all TF patches loaded from the same clip share this ID (the label too)
            examples, labels, index = examples.to(device), labels.to(device), index.to(device)
            idx_batch = torch.unique(index)
            counter = 0

            for i in range(len(idx_batch)): # Each iteration averages softmax predictions to get each clip prediction
                # process one evaluation clip
                num_examples = len(torch.nonzero(index==idx_batch[i]))
                output = model(examples[counter:counter+num_examples].unsqueeze(1))  # logits per patch

                loss = criterion(output, labels[counter:counter+num_examples])
                output = F.softmax(output, dim=1)    # class probabilities per patch
                output2 = torch.mean(output, dim=0)  # class probabilities per clip

                pred = output2.argmax()  # get the index of the max log-probability
                gt_label = labels[counter:counter + num_examples][0]

                # update count for correct (0 or 1), and total (1)
                correct[gt_label] += pred.eq(gt_label.view_as(pred)).sum().item()
                total[gt_label] += 1

                # compute unbalanced (micro) accuracy
                prec1, prec5 = accuracy_v2(output2.unsqueeze(0), labels[counter:counter+num_examples][0], top=[1, 5])
                test_loss.update(loss.item(), num_examples)
                top1.update(prec1.item(), num_examples)

                counter += num_examples

        # after all evaluation set, finalize per class accuracy, and compute balanced (macro) accuracy
        acc_per_class = 100*(correct/total)
        balanced_accuracy = acc_per_class.mean()

    return test_loss.avg, top1.avg, balanced_accuracy, acc_per_class


def eval_model_contrastive(args, model, model_head, device, test_loader, K):
    """
    Evaluation function for unsupervised learning: kNN evaluation
    :param args:
    :param model:
    :param model_head:
    :param device:
    :param test_loader:
    :param K:
    :return:
    """

    model.eval()
    model_head.eval()

    testFeatures = []
    testLabels = []
    totalSamples = 0
    correct = np.zeros((args['learn']['num_classes'],))
    correct_top3 = np.zeros((args['learn']['num_classes'],))
    total = np.zeros((args['learn']['num_classes'],))

    num_TF_patches = 0
    ## Loop to store features
    for batch_idx, (examples, labels, index) in enumerate(test_loader):
        examples, labels, index = examples.to(device), labels.to(device), index.to(device)
        idx_batch = torch.unique(index)
        counter = 0
        for i in range(len(idx_batch)):  ## Each iteration checks TF patches from the same clip
            num_examples = len(torch.nonzero(index == idx_batch[i]))
            if args['learn']['pretrained']:
                # model downloaded is less flexible
                output_ = model(examples[counter:counter + num_examples].unsqueeze(1))
            else:
                 _, output_ = model(examples[counter:counter + num_examples].unsqueeze(1))

            output = model_head(output_)
            num_TF_patches += output.size(0)
            testFeatures.append(output.data.cpu())
            testLabels.append(labels[counter:counter + num_examples].data.cpu())
            counter += num_examples


    ## Turn list of features and labels into Tensors!!
    testFeatures = torch.cat(testFeatures).t()
    testLabels = torch.cat(testLabels)
    ## k-nn eval
    top1 = 0.
    top3 = 0.

    with torch.no_grad():
        num_TF_patches2 = 0
        for batch_idx, (examples, labels, index) in enumerate(test_loader):
            examples, labels, index = examples.to(device), labels.to(device), index.to(device)
            idx_batch = torch.unique(index)
            counter = 0
            for i in range(len(idx_batch)):  ## Each iteration checks TF patches from the same clip
                num_examples = len(torch.nonzero(index == idx_batch[i]))
                if args['learn']['pretrained']:
                    # model downloaded is less flexible
                    output_ = model(examples[counter:counter + num_examples].unsqueeze(1))
                else:
                    _, output_ = model(examples[counter:counter + num_examples].unsqueeze(1))

                output = model_head(output_)
                gt_label = labels[counter:counter + num_examples][0].data.cpu()
                output = output.data.cpu()

                dist = torch.mm(output, testFeatures)
                idx2select_ = torch.ones(dist.size()[1])
                idx2select_[num_TF_patches2:num_TF_patches2 + num_examples] = 0
                idx2select = torch.where(idx2select_== 1)[0]
                dist_corrected = torch.index_select(dist, dim=1, index=idx2select) # dist without considering samples from current clip

                yd, yi = dist_corrected.topk(K, dim=1, largest=True, sorted=True)
                candidates = testLabels.view(1, -1).expand(num_examples, -1)
                retrieval = torch.gather(candidates, 1, yi)

                ## Get clip prediction by majority voting of each individual TF-patch dominant label

                labels2 = torch.zeros((retrieval.size(0),)).long()
                for kk in range(retrieval.size(0)):
                    values_, counts_ = torch.unique(retrieval[kk, :], return_counts=True)
                    labels2[kk] = values_[counts_.argmax()]

                values, counts = torch.unique(labels2, return_counts=True)
                clip_prediction = values[counts.argmax()]
                if len(values)>=3:
                    clip_prediction2 = values[torch.topk(counts, 3)[1]]
                elif len(values)==2:
                    clip_prediction2 = values[torch.topk(counts, 2)[1]]
                else:
                    clip_prediction2 = values[counts.argmax()]

                agreement = float((clip_prediction==gt_label).sum())
                agreement_top3 = float((clip_prediction2 == gt_label).sum())

                top1 = top1 + agreement
                top3 = top3 + agreement_top3

                ## balanced accuracy
                correct[gt_label] += agreement
                correct_top3[gt_label] += agreement_top3
                total[gt_label] += 1

                totalSamples += 1
                num_TF_patches2 += output.size(0)
                counter += num_examples

    top1 = 100*(top1/totalSamples)
    top3 = 100 * (top3 / totalSamples)
    acc_per_class = 100 * (correct / total)
    acc_per_class_top3 = 100 * (correct_top3 / total)
    balanced_accuracy = acc_per_class.mean()
    balanced_accuracy_top3 = acc_per_class_top3.mean()

    return top1, top3, balanced_accuracy, balanced_accuracy_top3, acc_per_class