import time
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt


def plotAUC(fpr,tpr,auc):
    plt.figure()
    plt.plot(fpr, tpr, 'k-',
             label='Mean ROC(area = %0.2f)' % (auc), lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('test.jpg',sep='')


def get_mean_AUC(fpr_list, tpr_list):
    '''
    :param fpr_list: 是一个list，list中每个元素是一个向量，这个向量是一次验证试验计算得到的不同阈值下的fpr；
    :param tpr_list: 是一个list，list中每个元素是一个向量，这个向量是一次验证试验计算得到的不同阈值下的tpr
    :return: 多次验证试验的平均预测auc，和平均fpr和平均tpr
    '''
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 1000)
    for i in range(len(fpr_list)):
        mean_tpr += interp(mean_fpr, fpr_list[i], tpr_list[i])
    mean_tpr /= len(fpr_list)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # plotAUC(mean_fpr, mean_tpr, mean_auc)
    return mean_auc, mean_fpr, mean_tpr


def eval_test_nn(gcn, classifier, test_loader, device, dataset, crit):
    gcn.eval()
    classifier.eval()
    correct = 0
    count = 0
    test_loss = 0.0
    data_list = []
    label_list = []
    for batch_ind, batch_target in test_loader:
        batch_target = batch_target.to(device)
        # forward
        emb = gcn(dataset)
        batch_output = classifier(emb, batch_ind)

        batch_output = batch_output.detach()
        # loss
        loss = crit(batch_output, batch_target)
        test_loss += loss.item() * batch_target.size(0)
        # prepare for auc
        output = batch_output.detach().cpu().numpy()
        target = batch_target.cpu().numpy()
        prob = list(output[:, 0])
        target = list(target)
        data_list.extend(prob)
        label_list.extend(target)
        # acc
        _, pred = batch_output.max(dim=1)
        correct += float(pred.eq(batch_target).sum().item())
        count += batch_target.size(0)
    acc = correct / count
    test_loss /= count
    # print(data_list)
    fpr, tpr, thresholds = roc_curve(label_list, data_list, pos_label=0)
    roc_auc = auc(fpr, tpr)

    return acc, test_loss, roc_auc


def eval_all_nn(gcn, classifier, device, dataset, gene_list,
             positive_train, positive_test, negative_train):
    gcn.eval()
    classifier.eval()
    tpr_list = []
    fpr_list = []
    all_rank = []
    used_link = positive_train + negative_train
    used_link_dict = {str(l[0])+"#"+str(l[1]): True for l in used_link}
    test_count = 0
    t1 = time.time()
    for test_omim, test_gene in positive_test:
        prob_list = []
        label_list = []
        # add ourself
        link = torch.LongTensor([[test_omim, test_gene]]).to(device)
        emb = gcn(dataset)
        output = classifier(emb, link)

        output = output.detach()
        prob = float(output.cpu()[0][0])
        prob_list.append(prob)
        label_list.append(1)
        pred_prob = prob
        # add other
        temp_link_list = []
        count = 0
        for gene in gene_list:
            link = [test_omim, gene]
            link_str = str(test_omim)+"#"+str(gene)
            # if used
            if link_str in used_link_dict:
                continue
            temp_link_list.append(link)
            count += 1
        if count > 0:
            link = torch.LongTensor(temp_link_list).to(device)
            emb = gcn(dataset)
            output = classifier(emb, link)

            output = output.detach().cpu().numpy()
            prob = list(output[:, 0])
            prob_list.extend(prob)
            label_list.extend([0 for _ in range(count)])

        # calculate auc
        fpr, tpr, thresholds = roc_curve(label_list, prob_list, pos_label=1)
        rank = sorted(prob_list)[::-1].index(pred_prob) + 1
        all_rank.append(rank)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        test_count += 1
        # if test_count % 10 == 0:
        #     t2 = time.time()
        #     print(t2 - t1, test_count)
        #     t1 = time.time()
    mean_auc, _, _ = get_mean_AUC(fpr_list, tpr_list)
    mean_rank = np.mean(np.array(all_rank))
    return mean_auc, mean_rank, fpr_list, tpr_list


def eval_test_dm(gcn, classifier, test_loader, device, dataset, crit):
    gcn.eval()
    classifier.eval()
    correct = 0
    count = 0
    test_loss = 0.0
    data_list = []
    label_list = []

    with torch.no_grad():
        # forward
        emb = gcn(dataset)
        for batch_ind, batch_target in test_loader:
            batch_target = batch_target.to(device)
            # distmult
            batch_output = classifier(emb, batch_ind)
            # loss
            loss = crit(batch_output, batch_target)
            test_loss += loss.item() * batch_target.size(0)
            # prepare for auc
            output = batch_output.detach().cpu().numpy()
            target = batch_target.cpu().numpy().astype(int)
            data_list.extend(list(output))
            label_list.extend(list(target))
            # acc
            pred = (output > 0.5).astype(int)
            # print(pred.item(), batch_target.item())
            correct += float((pred == target).sum())
            count += batch_target.size(0)
    acc = correct / count
    test_loss /= count
    # print(data_list)
    fpr, tpr, thresholds = roc_curve(label_list, data_list, pos_label=1)
    roc_auc = auc(fpr, tpr)

    return acc, test_loss, roc_auc


def eval_all_dm(gcn, classifier, device, dataset, gene_list,
             positive_train, positive_test, negative_train):
    gcn.eval()
    classifier.eval()
    tpr_list = []
    fpr_list = []
    all_rank = []
    used_link = positive_train + negative_train
    used_link_dict = {str(l[0]) + "#" + str(l[1]): True for l in used_link}
    test_count = 0
    t1 = time.time()
    with torch.no_grad():
        emb = gcn(dataset)
        for test_omim, test_gene in positive_test:
            prob_list = []
            label_list = []
            # add ourself
            link = torch.LongTensor([[test_omim, test_gene]]).to(device)
            if test_gene == -1:
                output = torch.tensor([0])
            else:
                output = classifier(emb, link)

            prob = float(output.cpu()[0])
            prob_list.append(prob)
            label_list.append(1)
            pred_prob = prob
            # add other
            temp_link_list = []
            count = 0
            invalid_index = []
            for gene in gene_list:
                link = [test_omim, gene]
                link_str = str(test_omim) + "#" + str(gene)
                # if used
                if link_str in used_link_dict:
                    continue
                temp_link_list.append(link)
                if gene == -1:
                    invalid_index.append(count)
                count += 1
            if count > 0:
                link = torch.LongTensor(temp_link_list).to(device)
                # distmult
                emb = gcn(dataset)
                output = classifier(emb, link)

                output = output.detach().cpu().numpy()
                if len(invalid_index) > 0:
                    invalid_index = np.array(invalid_index)
                    output[invalid_index] = 0
                prob = list(output)
                prob_list.extend(prob)
                label_list.extend([0 for _ in range(count)])

            # calculate auc
            fpr, tpr, thresholds = roc_curve(label_list, prob_list, pos_label=1)
            rank = sorted(prob_list)[::-1].index(pred_prob) + 1
            all_rank.append(rank)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            test_count += 1
            # if test_count % 10 == 0:
            #     t2 = time.time()
            #     print(t2 - t1, test_count)
            #     t1 = time.time()
    mean_auc, _, _ = get_mean_AUC(fpr_list, tpr_list)
    mean_rank = np.mean(np.array(all_rank))
    return mean_auc, mean_rank, fpr_list, tpr_list