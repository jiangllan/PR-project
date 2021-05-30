import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def load_ground_true(test_csv, include_self=True):

    ground_truth = np.zeros((len(test_csv), len(test_csv)))
    for i in range(len(test_csv)):
        for j in range(len(test_csv.iloc[i, :].target)):
            target_index = test_csv[test_csv.posting_id == test_csv.iloc[i, :].target[j]].index.tolist()[0]
            ground_truth[i, target_index] = 1

    if not include_self:
        mask = 1 - np.diag(np.ones(ground_truth.shape[0]))
        ground_truth *= mask

    return ground_truth.astype(int)


def main():

    test_csv_file = '../../data/new_split_data/new_test.csv'
    test_csv = pd.read_csv(test_csv_file)
    test_csv['image'] = '../../data/' + 'train_images/' + test_csv['image']
    tmp = test_csv.groupby('label_group').posting_id.agg('unique').to_dict()
    test_csv['target'] = test_csv.label_group.map(tmp)

    ground_truth = load_ground_true(test_csv, include_self=True)

    model_pred_file = '../../log/image-only/resnet50_pretrained/resnet50_pretrained.pickle'
    with open(os.path.join(model_pred_file), "rb") as f:
        pred = np.array(pickle.load(f)).astype(int)

    ### Total wrong pred num
    total_wrong_num = 0
    for i in range(ground_truth.shape[0]):
        total_wrong_num += np.sum(np.abs(ground_truth[i] - pred[i]))

    print(total_wrong_num)

    ## Ground Truth num is 2 and Pred not right
    ground_truth_num_2_index = []
    wrong_pred_num_gt_2 = 0
    for i in range(ground_truth.shape[0]):
        if np.sum(ground_truth[i]) == 2:
            ground_truth_num_2_index.append(i)
            wrong_pred_num_gt_2 += np.sum(np.abs(ground_truth[i] - pred[i]))

    print(wrong_pred_num_gt_2)
    print(len(ground_truth_num_2_index))
    print('\n')

    for i in range(len(ground_truth_num_2_index)):
        if i < 20:
            sample_index = ground_truth_num_2_index[i]
            gt_match_sample = np.where(ground_truth[sample_index] == 1)[0][:]
            pred_match_sample = np.where(pred[sample_index] == 1)[0][:]

            plt.figure()
            for j in range(len(gt_match_sample)):
                img = Image.open(test_csv.iloc[gt_match_sample[j],].image).convert('RGB')
                ax = plt.subplot(2, max(len(gt_match_sample), len(pred_match_sample)), j + 1)
                ax.set_title("Ground Truth")
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])
                plt.imshow(img)

            for j in range(len(pred_match_sample)):
                img = Image.open(test_csv.iloc[pred_match_sample[j],].image).convert('RGB')
                ax = plt.subplot(2, max(len(gt_match_sample), len(pred_match_sample)),
                            max(len(gt_match_sample), len(pred_match_sample)) + j + 1)
                ax.set_title("Prediction")
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])
                plt.imshow(img)

            plt.savefig('/home/jhj/target_num2_{}.jpg'.format(i))


    ### Ground Truth num 2 < < 10 and Pred not right
    ground_truth_num_2_10_index = []
    wrong_pred_num_gt_2_10 = 0
    for i in range(ground_truth.shape[0]):
        if 2 < np.sum(ground_truth[i]) < 10:
            ground_truth_num_2_10_index.append(i)
            wrong_pred_num_gt_2_10 += np.sum(np.abs(ground_truth[i] - pred[i]))

    print(wrong_pred_num_gt_2_10)
    print(len(ground_truth_num_2_10_index))
    print('\n')

    for i in range(len(ground_truth_num_2_10_index)):
        if i < 20:
            sample_index = ground_truth_num_2_10_index[i]
            gt_match_sample = np.where(ground_truth[sample_index] == 1)[0][:]
            pred_match_sample = np.where(pred[sample_index] == 1)[0][:]

            plt.figure()
            for j in range(len(gt_match_sample)):
                img = Image.open(test_csv.iloc[gt_match_sample[j],].image).convert('RGB')
                ax = plt.subplot(2, max(len(gt_match_sample), len(pred_match_sample)), j + 1)
                if j == 0:
                    ax.set_title("Ground Truth")
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])
                plt.imshow(img)

            for j in range(len(pred_match_sample)):
                img = Image.open(test_csv.iloc[pred_match_sample[j],].image).convert('RGB')
                ax = plt.subplot(2, max(len(gt_match_sample), len(pred_match_sample)),
                            max(len(gt_match_sample), len(pred_match_sample)) + j + 1)
                if j == 0:
                    ax.set_title("Prediction")
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])
                plt.imshow(img)

            plt.savefig('/home/jhj/target_num210_{}.jpg'.format(i))

    ### Ground Truth num == 10 and Pred not right
    ground_truth_num_10_index = []
    wrong_pred_num_gt_10 = 0
    for i in range(ground_truth.shape[0]):
        if np.sum(ground_truth[i]) == 10:
            ground_truth_num_10_index.append(i)
            wrong_pred_num_gt_10 += np.sum(np.abs(ground_truth[i] - pred[i]))

    print(wrong_pred_num_gt_10)
    print(len(ground_truth_num_10_index))

    for i in range(len(ground_truth_num_10_index)):
        if i < 20:
            sample_index = ground_truth_num_10_index[i]
            gt_match_sample = np.where(ground_truth[sample_index] == 1)[0][:]
            pred_match_sample = np.where(pred[sample_index] == 1)[0][:]

            plt.figure()
            for j in range(len(gt_match_sample)):
                img = Image.open(test_csv.iloc[gt_match_sample[j],].image).convert('RGB')
                ax = plt.subplot(2, max(len(gt_match_sample), len(pred_match_sample)), j + 1)
                if j == 0:
                    ax.set_title("Ground Truth")
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])
                plt.imshow(img)

            for j in range(len(pred_match_sample)):
                img = Image.open(test_csv.iloc[pred_match_sample[j],].image).convert('RGB')
                ax = plt.subplot(2, max(len(gt_match_sample), len(pred_match_sample)),
                            max(len(gt_match_sample), len(pred_match_sample)) + j + 1)
                if j == 0:
                    ax.set_title("Prediction")
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])
                plt.imshow(img)

            plt.savefig('/home/jhj/target_num10_{}.jpg'.format(i))








if __name__ == '__main__':
    main()