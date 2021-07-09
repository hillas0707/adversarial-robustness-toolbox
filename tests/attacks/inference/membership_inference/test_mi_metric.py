"""
Black- box access to ML model can be used to infer if a specific data item was in the model training set.
This could pose a significant security and privacy risk!
An extension of Fano’s inequality shows that the probability of success for a membership inference attack on ML model
can be bounded by an expression that depends on the mutual information between its inputs and its activations and/or outputs.

This is a set of experiments to test if there is a correlation between the accuracy of a membership inference attack
and the mutual information of the training data of the model and the model's predictions on the training data.
Paper:
(1) An Extension of Fano’s Inequality for Characterizing Model Susceptibility to Membership Inference Attacks - https://arxiv.org/pdf/2009.08097.pdf

As for now, we have:
- 2 attacks (both from ART)- black box, label only decision boundary
- 3 datasets-
    - german https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/
    - landsat https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/
    - loan
- 4 ML models (all from scikit learn)
    - logistic regression
    - decision tree
    - MLP
    - random forest

implemented by Hilla Schefler and Maya Avidor, as part of Industrial Project course 234313, CS department, Technion.
Hilla - hillas@campus.technion.ac.il
Maya - mayaavidor1@gmail.com
"""

from art.metrics.privacy.MIestimator import MI as mi
from art.attacks.inference.membership_inference.black_box import MembershipInferenceBlackBox
from art.attacks.inference.membership_inference.label_only_boundary_distance import LabelOnlyDecisionBoundary
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from scipy.stats import binom
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from math import comb, log
import pandas as pd
import pickle


class DataSetRecord:
    def __init__(self, x_train, x_test, y_train, y_test, name):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.name = name


class MIAResultRecord:
    def __init__(self, dataset, model, attack_name, attack_accuracy):
        self.dataset = dataset
        self.model = model
        self.attack_name = attack_name
        self.attack_accuracy = attack_accuracy


class Model:
    def __init__(self, classifier, model_name):
        self.classifier = classifier
        self.model_name = model_name


def collect_data_and_pickle(attacks_results, train_accuracy, test_accuracy, y_pred_train, y_pred_test):
    """
    Step 6- Pickling all data calculated from the attacks (on a specific dataset and a specific model).
    Calculating mi (different variation) with multiple p norm and k values.
    @param attacks_results: MIAResultRecord
    @param train_accuracy: model's accuracy on train data. float
    @param test_accuracy: model's accuracy on train data. float
    @param y_pred_train: model's probabilistic predictions on train data.
            shape- Nxd where N is the nuber of samples and d is the number of classes
    @param y_pred_test: model's probabilistic predictions on test data.
            shape- Nxd where N is the nuber of samples and d is the number of classes
    @return:
    """
    K = [1, 3, 5, 11]
    P = [float('inf'), 2.0, 1.0]
    x_train = attacks_results[0].dataset.x_train
    x_test = attacks_results[0].dataset.x_test
    y_train = [[float(i)] for i in attacks_results[0].dataset.y_train]
    y_test = [[float(i)] for i in attacks_results[0].dataset.y_test]

    size_of_train_data = x_train.shape[0]

    # 2 different types of 1-dim vector for Y, maybe increases correlation of attack acc and MI
    Y_pred_of_true_label = [[y_pred_train[i][int(y_train[i][0])]] for i in range(len(y_train))]
    Y_pred_max = [[max(row)] for row in y_pred_train]

    for k in K:
        for p in P:
            mi_train_pred_of_true_label = mi.mi_Kraskov_HnM(x_train, Y_pred_of_true_label, k=k, p_x=p, p_y=p)
            mi_train_pred_max = mi.mi_Kraskov_HnM(x_train, Y_pred_max, k=k, p_x=p, p_y=p)
            mi_train = mi.mi_Kraskov_HnM(x_train, y_pred_train, k=k, p_x=p, p_y=p)
            mi_train_true_labels = mi.mi_Kraskov_HnM(x_train, y_train, k=k, p_x=p, p_y=p)
            mi_test = mi.mi_Kraskov_HnM(x_test, y_pred_test, k=k, p_x=p, p_y=p)
            mi_test_true_labels = mi.mi_Kraskov_HnM(x_test, y_test, k=k, p_x=p, p_y=p)
            Print(f"\t\tk={k}, p={p}, mi={mi_train}")
            entropy_orig = mi.entropy(x_train, k=k)
            entropy_p = mi.entropy(x_train, k=k, p=p)
            entropy_p_wc = mi.entropy_with_correction(x_train, k=k, p=p)

            alpha = int(size_of_train_data / 2)
            const = log(sum([comb(size_of_train_data, i) for i in range(alpha + 1)]))
            LB_orig = (entropy_orig - mi_train - 1 - const) / (size_of_train_data - const)
            LB_p = (entropy_p - mi_train - 1 - const) / (size_of_train_data - const)
            LB_p_wc = (entropy_p_wc - mi_train - 1 - const) / (size_of_train_data - const)

            data = {"model": attacks_results[0].model.model_name, "data set": attacks_results[0].dataset.name, "k": k,
                    "p": p,
                    "size": size_of_train_data, "model accuracy- train": train_accuracy,
                    "model acc, test": test_accuracy, "mi train": mi_train, "mi test": mi_test,
                    "mi train true labels": mi_train_true_labels, "mi test true labels": mi_test_true_labels,
                    "mi train pred of true label": mi_train_pred_of_true_label, "mi train pred max": mi_train_pred_max,
                    "entropy- orig": entropy_orig, "entropy- p": entropy_p, "entropy- p with correction": entropy_p_wc,
                    "LB on prob of attack making more than 0.5|D| mistakes, orig": LB_orig, "LB on prob.. , p": LB_p,
                    "LB on prob.., p with correction": LB_p_wc}
            for attacks_res in attacks_results:
                rec = {f"attack acc {attacks_res.attack_name}": attacks_res.attack_accuracy}
                data.update(rec)
            with open("picklefile", 'ab') as pf:
                pickle.dump(data, pf)


def run_attack(dataset, model, attack_name):
    """
    Step 5- Run membership inference attack and return results.
    @param dataset: DataSetRecord
    @param model: Model
    @param attack_name:
    @return: MIAResultRecord
    """
    Print(f"\trunning {attack_name} attack:")

    ## Naming of vars explained:
    ## x/ y always indicates data and labels accordingly.
    ## First train\ test indicates if the model was trained on this data or not.
    ## Second train\ test indicates if the bb attack was trained on this data or not. bb atack trains on both data that was
    # used to train the model AND data that was NOT used to train the model
    x_train_attack_train, x_train_attack_test, y_train_attack_train, y_train_attack_test = train_test_split(
        dataset.x_train,
        dataset.y_train,
        test_size=0.5,
        shuffle=False)
    x_test_attack_train, x_test_attack_test, y_test_attack_train, y_test_attack_test = train_test_split(dataset.x_test,
                                                                                                        dataset.y_test,
                                                                                                        test_size=0.5,
                                                                                                        shuffle=False)
    classifier = ScikitlearnClassifier(model.classifier)
    if attack_name == "black box":
        attack = MembershipInferenceBlackBox(classifier, attack_model_type='rf')
        attack.fit(x_train_attack_train, y_train_attack_train, x_test_attack_train, y_test_attack_train)
    elif attack_name == "label only decision boundary":
        attack = LabelOnlyDecisionBoundary(classifier)
        attack.calibrate_distance_threshold(x_train_attack_train, y_train_attack_train, x_test_attack_train,
                                            y_test_attack_train)
    else:
        assert False, f"{attack_name} attack is not supported"

    x_attack_test = np.concatenate((x_train_attack_test, x_test_attack_test))
    y_attack_test = np.concatenate((y_train_attack_test, y_test_attack_test))
    assert x_attack_test.shape[0] == y_attack_test.shape[0]
    ones = np.ones(x_train_attack_test.shape[0])
    zeros = np.zeros(x_test_attack_test.shape[0])
    true_membership = np.concatenate((ones, zeros))

    inferred_membership = attack.infer(x_attack_test, y_attack_test)
    TN, FP, FN, TP = confusion_matrix(true_membership, inferred_membership).ravel()
    Print(f"\t\tconfusion mat of {attack_name} inference attack:")
    Print("\t\tTN ", TN)
    Print("\t\tFP ", FP)
    Print("\t\tFN ", FN)
    Print("\t\tTP ", TP)
    attack_accuracy = (TP + TN) / (TN + FP + FN + TP)
    assert attack_accuracy == accuracy_score(true_membership, inferred_membership)
    mistake_prob = (FP + FN) / (TN + FP + FN + TP)
    Print(f"\t\t{attack_name} accuracy ", attack_accuracy)
    res = MIAResultRecord(dataset, model, attack_name, attack_accuracy)
    return res


def run_inference_attacks(dataset, model):
    """
    Step 4- Run model on train and test data to get model's probabilistic predictions, call attacks and collect results and data.
    @param dataset: DataSetRecord
    @param model: Model
    @return:
    """
    attacks = ["black box", "label only decision boundary"]
    y_pred_train = model.classifier.predict_proba(dataset.x_train)
    y_pred_test = model.classifier.predict_proba(dataset.x_test)
    accuracy_train = accuracy_score(dataset.y_train, model.classifier.predict(dataset.x_train))
    accuracy_test = accuracy_score(dataset.y_test, model.classifier.predict(dataset.x_test))
    Print("\taccuracy of model's predictions on train set: ", accuracy_train)
    Print("\taccuracy of model's predictions on test set: ", accuracy_test)

    attacks_results = []
    for attack in attacks:
        attacks_results.append(run_attack(dataset, model, attack))
    collect_data_and_pickle(attacks_results, accuracy_train, accuracy_test, y_pred_train, y_pred_test)


def run_logistic_regression(dataset):
    """
    Step 3- Creating classifier and training it
    @param dataset: DataSetRecord
    @return:
    """
    Print(f'\t##running inference attacks - Logistic Regression')
    clf = LogisticRegression(random_state=42, max_iter=150)
    clf.fit(dataset.x_train, dataset.y_train)
    lr = Model(clf, "logistic regression")
    run_inference_attacks(dataset, lr)


def run_decision_tree(dataset):
    """
    Step 3- Creating classifier and training it
    @param dataset: DataSetRecord
    @return:
    """
    Print(f'\t##running inference attacks - Decision Tree')
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(dataset.x_train, dataset.y_train)
    dt = Model(clf, "decision tree")
    run_inference_attacks(dataset, dt)


def run_MLP(dataset):
    """
    Step 3- Creating classifier and training it
    @param dataset: DataSetRecord
    @return:
    """
    Print(f'\t##running black box attack - MLP')
    clf = MLPClassifier(random_state=0)
    clf.fit(dataset.x_train, dataset.y_train)
    mlp = Model(clf, "MLP")
    run_inference_attacks(dataset, mlp)


def run_random_forest(dataset):
    """
    Step 3- Creating classifier and training it
    @param dataset: DataSetRecord
    @return:
    """
    Print(f'\t##running black box attack - Random Forest')
    clf = RandomForestClassifier(random_state=0)
    clf.fit(dataset.x_train, dataset.y_train)
    rf = Model(clf, "random forest")
    run_inference_attacks(dataset, rf)


def run_all_models(dataset):
    """
    Step 2- Calling experiments for each model
    @param dataset: DataSetRecord
    @return:
    """
    Print("\ttrain data shape: ", dataset.x_train.shape)
    Print("\ttest data shape: ", dataset.x_test.shape)
    Print("\ttrain labels shape: ", dataset.y_train.shape)
    Print("\ttest labels shape: ", dataset.y_test.shape)

    run_logistic_regression(dataset)
    run_decision_tree(dataset)
    run_MLP(dataset)
    run_random_forest(dataset)


def run_all_tests_german():
    """
    Step 1- german dataset. Preprocess data (dividing to train- test)
    @return:
    """
    Print("testing with german data...")
    german = np.loadtxt('german.data-numeric.csv', delimiter=',')
    german_normalized = preprocessing.MinMaxScaler().fit_transform(german)
    x = german_normalized[:, :-1]  # deleting labels from samples
    y = german_normalized[:, -1]  # labels of samples
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

    german = DataSetRecord(x_train, x_test, y_train, y_test, "german")
    run_all_models(german)


def run_all_tests_landsat():
    """
    Step 1- landsat dataset. Preprocess data (dividing to train- test)
    @return:
    """
    Print("testing with landsat data...")
    landsat_train = np.loadtxt('sat.trn', delimiter=',')
    landsat_test = np.loadtxt('sat.tst', delimiter=',')
    y_train = landsat_train[:, -1]
    y_test = landsat_test[:, -1]
    landsat_train_normalized = preprocessing.MinMaxScaler().fit_transform(landsat_train)
    landsat_test_normalized = preprocessing.MinMaxScaler().fit_transform(landsat_test)
    x_train = landsat_train_normalized[:, :-1]  # deleting labels from samples- train set
    x_test = landsat_test_normalized[:, :-1]  # deleting labels from samples- test set
    num_of_samples = 800
    x_train = x_train[:num_of_samples]
    x_test = x_test[:num_of_samples]
    y_train = y_train[:num_of_samples]
    y_test = y_test[:num_of_samples]
    label_encoder = preprocessing.LabelEncoder()
    y = np.concatenate((y_train, y_test))
    label_encoder.fit(y)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)
    landsat = DataSetRecord(x_train, x_test, y_train, y_test, "landsat")
    run_all_models(landsat)


def run_all_tests_loan():
    """
    Step 1- loan dataset. Preprocess data (dividing to train- test)
    @return:
    """
    Print("testing with loan data...")
    loan = np.loadtxt('loan_train_from_abigail.csv', delimiter=',')
    loan_normalized = preprocessing.MinMaxScaler().fit_transform(loan)
    loan_normalized = loan_normalized[:2000]
    x = loan_normalized[:, :-1]  # deleting labels from samples
    y = loan_normalized[:, -1]  # labels of samples
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
    loan = DataSetRecord(x_train, x_test, y_train, y_test, 'loan')
    run_all_models(loan)


def plot_and_print_correlation(x, y, models, datasets, markers, colors, title, x_label, y_label, filename, attack_type):
    """
    Ploting scatter graphs and saving in PNG format.
    @param x: List of x-axis values
    @param y: List of y-axis values
    @param models: List of ML models ordered as they appear in data points in x/ y
    @param datasets: List of datasets ordered as they appear in data points in x/ y
    @param markers: List of markers for scatter points, each model has different marker
    @param colors: List of colors for scatter points, each dataset has different color
    @param title: Title of graph. string
    @param x_label: Label for x- axis. string
    @param y_label: Label for y- axis. string
    @param filename: name of png. string
    @param attack_type: Name of attack. string
    @return:
    """
    assert len(models) == len(colors)
    assert len(datasets) == len(markers)
    plt.figure()
    for i in range(len(x)):
        j = i % len(models)
        l = int(i / (len(x) / len(datasets)))
        plt.scatter(x[i], y[i], label=f"{attack_type}, {datasets[l]} {models[j]}", color=colors[l], marker=markers[j])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(filename)
    plt.close()

    cor, p_val = pearsonr(x, y)
    Print(title)
    Print(f"perason correlation- {attack_type} attack: {cor}, p-value: {p_val}")


def unpickle_and_plot_results():
    """
    Step 7- Unpickling records from all runs, calculating correlation of different attribute to inference attack accuracy
    and plotting results.
    List of graphs:
    - attack accuracy as a function of lower bound of probability of attack making more than 0.5|D| mistakes (x2 attacks)
    - attack accuracy as a function of mi of training data of the model and model's probabilistic predictions on training data (x2 attacks)
    - attack accuracy as a function of mi of test data of the model and model's probabilistic predictions on test data (x2 attacks)
    - attack accuracy as a function of normalized mi: mi of train data and model's probabilistic predictions divided by
        mi of of train data and true labels of train data (x2 attacks)
    - attack accuracy as a function of normalized mi: mi of test data and model's probabilistic predictions divided by
        mi of of test data and true labels of test data (x2 attacks)
    - attack accuracy as a function of mi of train data and probabilistic prediction of the correct class on train data
        (Y is 1-dim) (x2 attacks)
    - attack accuracy as a function of mi of train data and probabilistic prediction of the class with highest probability
        (Y is 1-dim) (x2 attacks)
    - attack accuracy as a function of the difference of mi of train data and probabilistic predictions and mi of test
        data and probabilistic predictions (x2 attacks)
    - attack accuracy as a function of train test gap- difference of model's train accuracy and test accuracy
    @return:
    """
    data = []
    with open("picklefile", 'rb') as pf:
        try:
            while True:
                data.append(pickle.load(pf))
        except EOFError:
            pass
    df = pd.DataFrame(data)
    pd.DataFrame.to_csv(df, "res.csv")
    K = df["k"].unique()
    P = df["p"].unique()
    datasets = df["data set"].unique()
    models = df["model"].unique()
    colors = ['r', 'g', 'b']
    markers = [".", "v", "p", "+"]
    assert len(models) == len(markers), "markers number doesn't match to number of models"
    assert len(datasets) == len(colors), "colors number doesn't match to number of data sets"
    models = ["lr", "dt", "mlp", "rf"]
    for k in K:
        LB_ = df.loc[df["k"] == k]
        mi_ = df.loc[df["k"] == k]
        acc_ = df.loc[df["k"] == k]
        for p in P:
            LB_p = LB_.loc[LB_["p"] == p, "LB on prob.., p with correction"].tolist()
            acc_bb = acc_.loc[acc_["p"] == p, "attack acc black box"].tolist()
            acc_ldb = acc_.loc[acc_["p"] == p, "attack acc label only decision boundary"].tolist()
            mi_train_p = mi_.loc[mi_["p"] == p, "mi train"].tolist()
            mi_test_p = mi_.loc[mi_["p"] == p, "mi test"].tolist()
            mi_train_true_labels_p = mi_.loc[mi_["p"] == p, "mi train true labels"].tolist()
            mi_test_true_labels_p = mi_.loc[mi_["p"] == p, "mi test true labels"].tolist()
            mi_train_pred_true_label_p = mi_.loc[mi_["p"] == p, "mi train pred of true label"].tolist()
            mi_train_pred_max_p = mi_.loc[mi_["p"] == p, "mi train pred max"].tolist()

            assert len(mi_train_pred_true_label_p) == len(mi_train_pred_max_p) == len(mi_test_p) == len(
                mi_train_p) == len(acc_ldb) == len(acc_bb) == len(LB_p) == len(mi_train_true_labels_p) == len(
                mi_test_true_labels_p)

            plot_and_print_correlation(LB_p, acc_bb, models, datasets, markers, colors,
                                       title=f"LB VS. bb attack accuracy- p={p}, k={k}", x_label="LB",
                                       y_label="attack accuracy", filename=f'LB VS bb attack acc p={p} k={k}.png',
                                       attack_type="bb")
            plot_and_print_correlation(LB_p, acc_ldb, models, datasets, markers, colors,
                                       title=f"LB VS. ldb attack accuracy- p={p}, k={k}", x_label="LB",
                                       y_label="attack accuracy", filename=f'LB VS ldb attack acc p={p} k={k}.png',
                                       attack_type="ldb")
            plot_and_print_correlation(mi_train_p, acc_bb, models, datasets, markers, colors,
                                       title=f"MI VS. bb attack accuracy- p={p}, k={k}", x_label="MI",
                                       y_label="attack accuracy", filename=f'MI VS bb attack acc p={p} k={k}.png',
                                       attack_type="bb")
            plot_and_print_correlation(mi_train_p, acc_ldb, models, datasets, markers, colors,
                                       title=f"MI VS. ldb attack accuracy- p={p}, k={k}", x_label="MI",
                                       y_label="attack accuracy", filename=f'MI VS ldb attack acc p={p} k={k}.png',
                                       attack_type="ldb")
            plot_and_print_correlation(mi_test_p, acc_bb, models, datasets, markers, colors,
                                       title=f"MI (test) VS. bb attack accuracy- p={p}, k={k}", x_label="MI",
                                       y_label="attack accuracy",
                                       filename=f'MI (test) VS bb attack acc p={p} k={k}.png',
                                       attack_type="bb")
            plot_and_print_correlation(mi_test_p, acc_ldb, models, datasets, markers, colors,
                                       title=f"MI(test) VS. ldb attack accuracy- p={p}, k={k}", x_label="MI",
                                       y_label="attack accuracy",
                                       filename=f'MI (test) VS ldb attack acc p={p} k={k}.png',
                                       attack_type="ldb")
            mi_train_normalized = [x / y for x, y in zip(mi_train_p, mi_train_true_labels_p)]
            plot_and_print_correlation(mi_train_normalized, acc_bb, models, datasets, markers, colors,
                                       title=f"MI (normalized) VS. bb attack accuracy- p={p}, k={k}", x_label="MI",
                                       y_label="attack accuracy",
                                       filename=f'MI (normalized) VS bb attack acc p={p} k={k}.png',
                                       attack_type="bb")
            plot_and_print_correlation(mi_train_normalized, acc_ldb, models, datasets, markers, colors,
                                       title=f"MI (normalized) VS. ldb attack accuracy- p={p}, k={k}", x_label="MI",
                                       y_label="attack accuracy",
                                       filename=f'MI (normalized) VS ldb attack acc p={p} k={k}.png',
                                       attack_type="ldb")
            mi_test_normalized = [x / y for x, y in zip(mi_test_p, mi_test_true_labels_p)]
            plot_and_print_correlation(mi_test_normalized, acc_bb, models, datasets, markers, colors,
                                       title=f"MI (test, normalized) VS. bb attack accuracy- p={p}, k={k}",
                                       x_label="MI",
                                       y_label="attack accuracy",
                                       filename=f'MI (test, normalized) VS bb attack acc p={p} k={k}.png',
                                       attack_type="bb")
            plot_and_print_correlation(mi_test_normalized, acc_ldb, models, datasets, markers, colors,
                                       title=f"MI(test, normalized) VS. ldb attack accuracy- p={p}, k={k}",
                                       x_label="MI",
                                       y_label="attack accuracy",
                                       filename=f'MI (test, normalized) VS ldb attack acc p={p} k={k}.png',
                                       attack_type="ldb")
            plot_and_print_correlation(mi_train_pred_true_label_p, acc_bb, models, datasets, markers, colors,
                                       title=f"MI_V2 (Y = pred. of model on correct class) VS. bb attack accuracy- p={p}, k={k}",
                                       x_label="MI",
                                       y_label="attack accuracy", filename=f'MI_V2 VS bb attack acc p={p} k={k}.png',
                                       attack_type="bb")
            plot_and_print_correlation(mi_train_pred_true_label_p, acc_ldb, models, datasets, markers, colors,
                                       title=f"MI_V2 (Y = pred. of model on correct class) VS. ldb attack accuracy- p={p}, k={k}",
                                       x_label="MI",
                                       y_label="attack accuracy", filename=f'MI_V2 VS ldb attack acc p={p} k={k}.png',
                                       attack_type="ldb")
            plot_and_print_correlation(mi_train_pred_max_p, acc_bb, models, datasets, markers, colors,
                                       title=f"MI_V3 (Y = pred. of model on highest class) VS. bb attack accuracy- p={p}, k={k}",
                                       x_label="MI",
                                       y_label="attack accuracy", filename=f'MI_V3 VS bb attack acc p={p} k={k}.png',
                                       attack_type="bb")
            plot_and_print_correlation(mi_train_pred_max_p, acc_ldb, models, datasets, markers, colors,
                                       title=f"MI_V3 (Y = pred. of model on highest class) VS. ldb attack accuracy- p={p}, k={k}",
                                       x_label="MI",
                                       y_label="attack accuracy", filename=f'MI_V3 VS ldb attack acc p={p} k={k}.png',
                                       attack_type="ldb")
            mi_train_test_diff = [x - y for x, y in zip(mi_train_p, mi_test_p)]
            plot_and_print_correlation(mi_train_test_diff, acc_bb, models, datasets, markers, colors,
                                       title=f"MI train test diff. VS. bb attack accuracy- p={p}, k={k}",
                                       x_label="MI diff",
                                       y_label="attack accuracy",
                                       filename=f'MI train test diff VS bb attack acc p={p} k={k}.png',
                                       attack_type="bb")
            plot_and_print_correlation(mi_train_test_diff, acc_ldb, models, datasets, markers, colors,
                                       title=f"MI train test diff. VS. ldb attack accuracy- p={p}, k={k}",
                                       x_label="MI diff",
                                       y_label="attack accuracy",
                                       filename=f'MI train test diff VS ldb attack acc p={p} k={k}.png',
                                       attack_type="ldb")

    df_ = df.loc[df["k"] == 1]
    df_ = df_.loc[df_["p"] == float('inf')]
    acc_train = df_["model accuracy- train"].tolist()
    acc_test = df_["model acc, test"].tolist()
    attack_acc_bb = df_["attack acc bb"].tolist()
    attack_acc_ldb = df_["attack acc ldb"].tolist()

    delta = [x - y for x, y in zip(acc_train, acc_test)]
    plot_and_print_correlation(delta, attack_acc_bb, models, datasets, markers, colors,
                               title=f"overfitting gap VS. bb attack accuracy",
                               x_label="overfit gap (train accuracy - test accuracy)",
                               y_label="attack accuracy",
                               filename=f'overfit gap VS bb attack acc.png',
                               attack_type="bb")
    plot_and_print_correlation(delta, attack_acc_ldb, models, datasets, markers, colors,
                               title=f"overfitting gap VS. ldb attack accuracy",
                               x_label="overfit gap (train accuracy - test accuracy)",
                               y_label="attack accuracy",
                               filename=f'overfit gap VS ldb attack acc.png',
                               attack_type="ldb")

    plt.figure()
    for i in range(len(delta)):
        j = i % len(markers)
        l = int(i / (len(delta) / len(datasets)))
        plt.scatter(delta[i], attack_acc_bb[i], color=colors[l], marker=markers[j])
        plt.scatter(delta[i], attack_acc_ldb[i], color=colors[l], marker=markers[j])
    plt.title(f"overfitting gap VS. both attacks accuracy")
    plt.xlabel("overfit gap (train accuracy - test accuracy)")
    plt.ylabel("attack accuracy")
    # plt.legend()
    plt.savefig(f'overfit gap VS both attacks acc.png')
    plt.close()
    cor, p = pearsonr(delta + delta, attack_acc_bb + attack_acc_ldb)
    Print(f"overfitting gap VS. both attacks accuracy")
    Print(f"perason correlation- both attacks: {cor}, p-value: {p}")


def main():
    Print("----- STARTING TESTS! ------")
    run_all_tests_german()
    run_all_tests_landsat()
    run_all_tests_loan()
    unpickle_and_plot_results()
    Print("----- DONE :) -----")


def Print(string, *args):
    print(string, *args)
    print(string, *args, file=output)
    # output.flush()


if __name__ == "__main__":
    output = open("log.txt", "a", 1)
    main()
    output.close()
