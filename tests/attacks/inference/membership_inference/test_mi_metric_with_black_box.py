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
from matplotlib import pyplot as plt
from math import comb, log
import pandas as pd
import pickle


## p_aplpha >= lower bound. p_alpha is the probability of MIA model making more then alpha prediction errors
def calc_LB_prob_MIA_errors(training_data_D, output_Y, size_of_data_set, Alphas, k, p):
    mi_D_Y = mi.mi_Kraskov_HnM(training_data_D, output_Y, k=k, p_x=p,
                               p_y=p)  # p_y =p ?????????????????????????????????????????????/
    # Print(f'\t\tMI(D,Y)={mi_D_Y}, k={k}, p={p}')
    entropy_D_orig = mi.entropy(training_data_D, k=k)
    entropy_D_p = mi.entropy(training_data_D, k=k, p=p)
    entropy_D_p_wc = mi.entropy_with_correction(training_data_D, k=k, p=p)
    p_alphas_orig = []
    p_alphas_p = []
    p_alphas_p_wc = []
    for alpha in Alphas:
        const = log(sum([comb(size_of_data_set, i) for i in range(alpha + 1)]))
        p_alphas_orig.append((entropy_D_orig - mi_D_Y - 1 - const) / (size_of_data_set - const))
        p_alphas_p.append((entropy_D_p - mi_D_Y - 1 - const) / (size_of_data_set - const))
        p_alphas_p_wc.append((entropy_D_p_wc - mi_D_Y - 1 - const) / (size_of_data_set - const))
    return p_alphas_orig, p_alphas_p, p_alphas_p_wc


'''
def bar_plot_mi(xticks, y, width, labels, legends, n_bars, title):
    fig, ax = plt.subplots()
    for i in range(n_bars):
        rect = ax.bar(xticks + width * (2 * i + 1 - n_bars) / 2, y[i], width=width, label=f'{legends[i]} norm')
        ax.bar_label(rect, padding=3)
    ax.set_ylabel(f'MI(D,Y)  ({title} data)')
    ax.set_title(f'MI({title}, {title} predictions) as a function of p norm and k #nbrs')
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    # plt.show()
    plt.savefig(f'MI(p,k) bar chart {title}.png')


def plot_mi(x, y, title, p):
    plt.figure()
    plt.title(f"MI({title}, {title} predictions) as a function of k (#nbrs)- {p} norm")
    plt.xlabel("k (#neighbors used by estimator)")
    plt.ylabel(f"MI(D,Y)  ({title} data)")
    plt.plot(x, y, "ob")
    # plt.show()
    plt.savefig(f'MI(k) {p} norm {title}.png')
'''


def plot_mi_as_func_of_of_k(x_train, x_test, K, P, model_name, data_set_name):
    assert len(x_train) == len(x_test) == len(P)
    num_of_norms = len(P)
    plt.figure()
    for i in range(num_of_norms):
        plt.plot(K, x_train[i], label=f'train ds, p={str(P[i])}')
        plt.plot(K, x_test[i], label=f'test ds, p={str(P[i])}')
    plt.title(f"MI(D,predictions) as a function of k (#nbrs)- {model_name}, {data_set_name}")
    plt.xlabel("k (#neighbors used by estimator)")
    plt.ylabel("MI(D,Y)")
    plt.legend()
    # plt.show()
    plt.savefig(f'MI(D,Y)(k), {model_name}, {data_set_name}.png')


# calculating the probability of making at least alpha prediction mistakes given the (empirical) probability of making one mistake (assuming independence- binomial dist.)
def calc_prob_of_alpha_mistakes(mistake_prob, alpha, size_of_data_set):
    # return sum(
    #    [comb(size_of_data_set, i) * (mistake_prob ** i) * ((1 - mistake_prob) ** (size_of_data_set - i)) for i in
    #     range(alpha, size_of_data_set + 1)])
    return sum([binom.pmf(i, size_of_data_set, mistake_prob) for i in range(alpha, size_of_data_set + 1)])


# alpha is the number of mistakes that attack makes
def plot_probabilities_as_func_of_alpha(LBs, p_alphas, Alphas, K, p, model_name, data_set_name, *args):
    plt.figure()
    plt.rcParams["axes.titlesize"] = 6
    plt.plot(Alphas, p_alphas, label='empirical p_alpha')
    for i in range(len(K)):
        plt.plot(Alphas, LBs[i], label=f'LB with k={K[i]}')
    plt.title(
        f"probability of MIA attack making alpha prediction errors: empirical estimation vs. lower bound with mi - {model_name}, {data_set_name}, p_norm={p}")
    plt.xlabel("alpha (number of prediction errors of MIA attack)")
    plt.ylabel(f'probability, {args[0]}')
    plt.legend()
    plt.savefig(f'prob(alpha), p={p}, {model_name}, {data_set_name}, {args[0]}.png')


def run_inference_attacks(x_train, x_test, y_train, y_test, model, attack_type="black box", rf=True):
    Print(f"\trunning {attack_type} attack:")
    model.fit(x_train, y_train)
    y_pred_train = model.predict_proba(x_train)
    y_pred_test = model.predict_proba(x_test)
    # y_pred_train = np.array([y_pred_train]).T
    # y_pred_test = np.array([y_pred_test]).T
    accuracy_train = accuracy_score(y_train, model.predict(x_train))
    accuracy_test = accuracy_score(y_test, model.predict(x_test))
    Print("\t\taccuracy of model's predictions on train set: ", accuracy_train)
    Print("\t\taccuracy of model's predictions on test set: ", accuracy_test)

    ## Naming of vars explained:
    ## x/ y always indicates data and labels accordingly.
    ## First train\ test indicates if the model was trained on this data or not.
    ## Second train\ test indicates if the bb attack was trained on this data or not. bb atack trains on both data that was
    # used to train the model AND data that was NOT used to train the model
    x_train_attack_train, x_train_attack_test, y_train_attack_train, y_train_attack_test = train_test_split(x_train,
                                                                                                            y_train,
                                                                                                            test_size=0.5,
                                                                                                            shuffle=False)
    x_test_attack_train, x_test_attack_test, y_test_attack_train, y_test_attack_test = train_test_split(x_test, y_test,
                                                                                                        test_size=0.5,
                                                                                                        shuffle=False)
    classifier = ScikitlearnClassifier(model)
    if attack_type == "black box":
        if rf:
            attack = MembershipInferenceBlackBox(classifier, attack_model_type='rf')
        else:
            attack = MembershipInferenceBlackBox(classifier)
        attack.fit(x_train_attack_train, y_train_attack_train, x_test_attack_train, y_test_attack_train)
    else:
        assert attack_type == "label only decision boundary", "attack type should be black box or label only decision boundary"
        attack = LabelOnlyDecisionBoundary(classifier)
        attack.calibrate_distance_threshold(x_train_attack_train, y_train_attack_train, x_test_attack_train,
                                            y_test_attack_train)

    x_attack_test = np.concatenate((x_train_attack_test, x_test_attack_test))
    y_attack_test = np.concatenate((y_train_attack_test, y_test_attack_test))
    # Print(x_attack_test.shape)
    # Print(y_attack_test.shape)
    assert x_attack_test.shape[0] == y_attack_test.shape[0]
    ones = np.ones(x_train_attack_test.shape[0])
    zeros = np.zeros(x_test_attack_test.shape[0])
    true_membership = np.concatenate((ones, zeros))
    # Print(true_membership.shape)
    # Print(y_attack_test.shape)

    inferred_membership = attack.infer(x_attack_test, y_attack_test)
    # Print(infered_membership.shape)
    TN, FP, FN, TP = confusion_matrix(true_membership, inferred_membership).ravel()
    Print(f"\t\tconfusion mat of {attack_type} inference attack:")
    Print("\t\tTN ", TN)
    Print("\t\tFP ", FP)
    Print("\t\tFN ", FN)
    Print("\t\tTP ", TP)
    attack_accuracy = (TP + TN) / (TN + FP + FN + TP)
    # attack_accuracy = accuracy_score(true_membership, inferred_membership)
    assert attack_accuracy == accuracy_score(true_membership, inferred_membership)
    mistake_prob = (FP + FN) / (TN + FP + FN + TP)
    Print(f"\t\t{attack_type} accuracy ", attack_accuracy)
    return y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy, mistake_prob


def analyze_logistic_regression(x_train, x_test, y_train, y_test, data_set_name):
    Print(f'\t##running inference attacks - Logistic Regression')
    clf = LogisticRegression(random_state=42, max_iter=150)
    y_pred_train_bb, y_pred_test_bb, accuracy_train_bb, accuracy_test_bb, attack_accuracy_bb, mistake_prob_bb = run_inference_attacks(
        x_train,
        x_test,
        y_train,
        y_test,
        clf, attack_type="black box")
   # y_pred_train_ldb, y_pred_test_ldb, accuracy_train_ldb, accuracy_test_ldb, attack_accuracy_ldb, mistake_prob_ldb = run_inference_attacks(
    #    x_train,
    #    x_test,
    #    y_train,
    #    y_test,
    #    clf, attack_type="label only decision boundary")
    #assert (y_pred_train_bb == y_pred_train_ldb).all()
    #assert (y_pred_test_bb == y_pred_test_ldb).all()
    #assert accuracy_test_bb == accuracy_test_ldb
    #assert accuracy_train_bb == accuracy_train_ldb

    # analyze(x_train, x_test, y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy, mistake_prob,
    # "Logistic Regression", data_set_name)

    #attack_accuracy = [attack_accuracy_bb, attack_accuracy_ldb]
    attack_accuracy = [attack_accuracy_bb, 0]
    #analyze2(x_train, x_test, y_pred_train_bb, y_pred_test_bb, accuracy_train_bb, accuracy_test_bb, attack_accuracy,
             #"Logistic Regression", data_set_name)

    analyze3(x_train, x_test, y_pred_train_bb, y_pred_test_bb, y_train, accuracy_train_bb, accuracy_test_bb, attack_accuracy,
             "Logistic Regression", data_set_name)


def analyze_decision_tree(x_train, x_test, y_train, y_test, data_set_name):
    Print(f'\t##running inference attacks - Decision Tree')
    clf = DecisionTreeClassifier(random_state=0)
    y_pred_train_bb, y_pred_test_bb, accuracy_train_bb, accuracy_test_bb, attack_accuracy_bb, mistake_prob_bb = run_inference_attacks(
        x_train,
        x_test,
        y_train,
        y_test,
        clf, attack_type="black box")
   # y_pred_train_ldb, y_pred_test_ldb, accuracy_train_ldb, accuracy_test_ldb, attack_accuracy_ldb, mistake_prob_ldb = run_inference_attacks(
   #     x_train,
   #     x_test,
   #     y_train,
   #     y_test,
   #     clf, attack_type="label only decision boundary")
   # assert (y_pred_train_bb == y_pred_train_ldb).all()
   # assert (y_pred_test_bb == y_pred_test_ldb).all()
   # assert accuracy_test_bb == accuracy_test_ldb
   # assert accuracy_train_bb == accuracy_train_ldb

    # analyze(x_train, x_test, y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy, mistake_prob,
    # "Decision Tree", data_set_name)

    #attack_accuracy = [attack_accuracy_bb, attack_accuracy_ldb]
    attack_accuracy = [attack_accuracy_bb, 0]

    #analyze2(x_train, x_test, y_pred_train_bb, y_pred_test_bb, accuracy_train_bb, accuracy_test_bb, attack_accuracy,
             #"Decision Tree", data_set_name)

    analyze3(x_train, x_test, y_pred_train_bb, y_pred_test_bb, y_train, accuracy_train_bb, accuracy_test_bb, attack_accuracy,
             "Decision Tree", data_set_name)


def analyze_sklearn_MLPClassifier(x_train, x_test, y_train, y_test, data_set_name):
    Print(f'\t##running black box attack - MLP')
    clf = MLPClassifier(random_state=0)
    y_pred_train_bb, y_pred_test_bb, accuracy_train_bb, accuracy_test_bb, attack_accuracy_bb, mistake_prob_bb = run_inference_attacks(
        x_train,
        x_test,
        y_train,
        y_test,
        clf, attack_type="black box")
   # y_pred_train_ldb, y_pred_test_ldb, accuracy_train_ldb, accuracy_test_ldb, attack_accuracy_ldb, mistake_prob_ldb = run_inference_attacks(
   #     x_train,
   #     x_test,
   #     y_train,
   #     y_test,
   #     clf, attack_type="label only decision boundary")
   # assert (y_pred_train_bb == y_pred_train_ldb).all()
   # assert (y_pred_test_bb == y_pred_test_ldb).all()
   # assert accuracy_test_bb == accuracy_test_ldb
   # assert accuracy_train_bb == accuracy_train_ldb

    # analyze(x_train, x_test, y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy, mistake_prob,
    # "MLP", data_set_name)

    #attack_accuracy = [attack_accuracy_bb, attack_accuracy_ldb]
    attack_accuracy = [attack_accuracy_bb, 0]

    #analyze2(x_train, x_test, y_pred_train_bb, y_pred_test_bb, accuracy_train_bb, accuracy_test_bb, attack_accuracy,
             #"MLP", data_set_name)

    analyze3(x_train, x_test, y_pred_train_bb, y_pred_test_bb, y_train, accuracy_train_bb, accuracy_test_bb, attack_accuracy,
             "MLP", data_set_name)


def analyze_RandomForestClassifier(x_train, x_test, y_train, y_test, data_set_name):
    Print(f'\t##running black box attack - Random Forest')
    clf = RandomForestClassifier(random_state=0)
    y_pred_train_bb, y_pred_test_bb, accuracy_train_bb, accuracy_test_bb, attack_accuracy_bb, mistake_prob_bb = run_inference_attacks(
        x_train,
        x_test,
        y_train,
        y_test,
        clf, attack_type="black box")
   # y_pred_train_ldb, y_pred_test_ldb, accuracy_train_ldb, accuracy_test_ldb, attack_accuracy_ldb, mistake_prob_ldb = run_inference_attacks(
   #     x_train,
   #     x_test,
   #     y_train,
   #     y_test,
   #     clf, attack_type="label only decision boundary")
   # assert (y_pred_train_bb == y_pred_train_ldb).all()
   # assert (y_pred_test_bb == y_pred_test_ldb).all()
   # assert accuracy_test_bb == accuracy_test_ldb
   # assert accuracy_train_bb == accuracy_train_ldb

    # analyze(x_train, x_test, y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy, mistake_prob,
    # "Random Forest", data_set_name)

   # attack_accuracy = [attack_accuracy_bb, attack_accuracy_ldb]
    attack_accuracy = [attack_accuracy_bb, 0]

    #analyze2(x_train, x_test, y_pred_train_bb, y_pred_test_bb, accuracy_train_bb, accuracy_test_bb, attack_accuracy,
             #"Random Forest", data_set_name)

    analyze3(x_train, x_test, y_pred_train_bb, y_pred_test_bb, y_train, accuracy_train_bb, accuracy_test_bb, attack_accuracy,
             "Random Forest", data_set_name)

def analyze3(x_train, x_test, y_pred_train, y_pred_test, y_train, accuracy_train, accuracy_test, attack_accuracy, model_name,
             data_set_name):
    K = [1, 3, 5, 11]
    P = [float('inf'), 2.0, 1.0]
    size_of_train_data = x_train.shape[0]

    Y = [[y_pred_train[i][int(y_train[i])]] for i in range(len(y_train))]

    for k in K:
        for p in P:
            mi_train = mi.mi_Kraskov_HnM(x_train, Y, k=k, p_x=p, p_y=p)
            Print(f"\t\tk={k}, p={p}, mi={mi_train}")
            entropy_orig = mi.entropy(x_train, k=k)
            entropy_p = mi.entropy(x_train, k=k, p=p)
            entropy_p_wc = mi.entropy_with_correction(x_train, k=k, p=p)

            alpha = int(size_of_train_data / 2)
            const = log(sum([comb(size_of_train_data, i) for i in range(alpha + 1)]))
            LB_orig = (entropy_orig - mi_train - 1 - const) / (size_of_train_data - const)
            LB_p = (entropy_p - mi_train - 1 - const) / (size_of_train_data - const)
            LB_p_wc = (entropy_p_wc - mi_train - 1 - const) / (size_of_train_data - const)

            data = {"model": model_name, "data set": data_set_name, "model accuracy- train": accuracy_train,
                    "model acc, test": accuracy_test, "attack acc bb": attack_accuracy[0],
                    "attack acc ldb": attack_accuracy[1], "mi": mi_train, "entropy- orig": entropy_orig,
                    "entropy- p": entropy_p, "entropy- p with correction": entropy_p_wc,
                    "LB on prob of attack making more than 0.5|D| mistakes, orig": LB_orig, "LB on prob.. , p": LB_p,
                    "LB on prob.., p with correction": LB_p_wc,
                    "k": k, "p": p, "size": size_of_train_data}
            with open("picklefile", 'ab') as pf:
                pickle.dump(data, pf)


def analyze2(x_train, x_test, y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy, model_name,
             data_set_name):
    K = [1, 3, 5, 11]
    P = [float('inf'), 2.0, 1.0]
    size_of_train_data = x_train.shape[0]
    for k in K:
        for p in P:
            mi_train = mi.mi_Kraskov_HnM(x_train, y_pred_train, k=k, p_x=p, p_y=p)
            Print(f"\t\tk={k}, p={p}, mi={mi_train}")
            entropy_orig = mi.entropy(x_train, k=k)
            entropy_p = mi.entropy(x_train, k=k, p=p)
            entropy_p_wc = mi.entropy_with_correction(x_train, k=k, p=p)

            alpha = int(size_of_train_data / 2)
            const = log(sum([comb(size_of_train_data, i) for i in range(alpha + 1)]))
            LB_orig = (entropy_orig - mi_train - 1 - const) / (size_of_train_data - const)
            LB_p = (entropy_p - mi_train - 1 - const) / (size_of_train_data - const)
            LB_p_wc = (entropy_p_wc - mi_train - 1 - const) / (size_of_train_data - const)

            data = {"model": model_name, "data set": data_set_name, "model accuracy- train": accuracy_train,
                    "model acc, test": accuracy_test, "attack acc bb": attack_accuracy[0],
                    "attack acc ldb": attack_accuracy[1], "mi": mi_train, "entropy- orig": entropy_orig,
                    "entropy- p": entropy_p, "entropy- p with correction": entropy_p_wc,
                    "LB on prob of attack making more than 0.5|D| mistakes, orig": LB_orig, "LB on prob.. , p": LB_p,
                    "LB on prob.., p with correction": LB_p_wc,
                    "k": k, "p": p, "size": size_of_train_data}
            with open("picklefile", 'ab') as pf:
                pickle.dump(data, pf)


def analyze(x_train, x_test, y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy, mistake_prob,
            model_name, data_set_name):
    K = [*range(1, 18, 2)]
    # K = [2, 3, 5, 10]
    P = [float('inf'), 2.0, 1.0]
    size_of_data_set = x_train.shape[0] + x_test.shape[0]
    # alpha denotes the number of errors that attack makes. see thm.1 in https://arxiv.org/pdf/2009.08097.pdf
    Alphas = [*range(1, size_of_data_set + 1)]
    mi_train = []
    mi_test = []
    LB_orig = []
    LB_p = []
    LB_p_wc = []
    p_alphas = [calc_prob_of_alpha_mistakes(mistake_prob, alpha, size_of_data_set) for alpha in Alphas]
    for p in P:
        mi_train.append([])
        mi_test.append([])
        LB_orig.append([])
        LB_p.append([])
        LB_p_wc.append([])
        for k in K:
            mi_train[-1].append(mi.mi_Kraskov_HnM(x_train, y_pred_train, k=k, p_x=p,
                                                  p_y=p))  # SHOULD WE USE P_Y=P??????????????????????????????
            mi_test[-1].append(mi.mi_Kraskov_HnM(x_test, y_pred_test, k=k, p_x=p, p_y=p))
            lb_orig, lb_p, lb_p_wc = calc_LB_prob_MIA_errors(x_train, y_pred_train, size_of_data_set, Alphas, k=k, p=p)
            LB_orig[-1].append(lb_orig)
            LB_p[-1].append(lb_p)
            LB_p_wc[-1].append(lb_p_wc)
        # plot_mi(K, mi_train[-1], title='train', p=p)
        # plot_mi(K, mi_test[-1], title='test', p=p)
        plot_probabilities_as_func_of_alpha(LB_orig[-1], p_alphas, Alphas, K, p, model_name, data_set_name,
                                            "LB with orig entropy est.")
        plot_probabilities_as_func_of_alpha(LB_p[-1], p_alphas, Alphas, K, p, model_name, data_set_name,
                                            "LB with p-norm etropy est.")
        plot_probabilities_as_func_of_alpha(LB_p_wc[-1], p_alphas, Alphas, K, p, model_name, data_set_name,
                                            "LB with p-norm with correction entropy est.")

    plot_mi_as_func_of_of_k(mi_train, mi_test, K, P, model_name, data_set_name)

    # np.save("mi_train", mi_train)
    # np.save("mi_test", mi_test)
    # mi_train = np.load("mi_train.npy")
    # mi_test = np.load("mi_test.npy")

    ''' staff for bar plots- for now not needed '''
    # xticks = np.arange(len(K))
    # K_strings = [f'k = {k}' for k in K]
    # bar_plot_mi(xticks, y=mi_train, width=0.25, labels=K_strings, legends=P, n_bars=len(P), title='train')
    # bar_plot_mi(xticks, y=mi_test, width=0.25, labels=K_strings, legends=P, n_bars=len(P), title='test')


def analyze_all_models(x_train, x_test, y_train, y_test, data_set_name):
    Print("\ttrain data shape: ", x_train.shape)
    Print("\ttest data shape: ", x_test.shape)
    Print("\ttrain labels shape: ", y_train.shape)
    Print("\ttest labels shape: ", y_test.shape)
    analyze_logistic_regression(x_train, x_test, y_train, y_test, data_set_name)
    analyze_decision_tree(x_train, x_test, y_train, y_test, data_set_name)
    analyze_sklearn_MLPClassifier(x_train, x_test, y_train, y_test, data_set_name)
    analyze_RandomForestClassifier(x_train, x_test, y_train, y_test, data_set_name)


def test_german_with_multiple_models():
    Print("testing with german data...")
    german = np.loadtxt('german.data-numeric.csv', delimiter=',')
    german_normalized = preprocessing.MinMaxScaler().fit_transform(german)
    x = german_normalized[:, :-1]  # deleting labels from samples
    y = german_normalized[:, -1]  # labels of samples
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
    analyze_all_models(x_train, x_test, y_train, y_test, "german")


def test_landsat_with_multiple_models():
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
    analyze_all_models(x_train, x_test, y_train, y_test, "landsat")


def test_landsat_manually():
    Print("testing with landsat data... MANUALLY")
    landsat_train = np.loadtxt('sat.trn', delimiter=',')
    landsat_test = np.loadtxt('sat.tst', delimiter=',')
    y_train = landsat_train[:, -1]
    y_test = landsat_test[:, -1]
    landsat_train_normalized = preprocessing.MinMaxScaler().fit_transform(landsat_train)
    landsat_test_normalized = preprocessing.MinMaxScaler().fit_transform(landsat_test)
    x_train = landsat_train_normalized[:, :-1]  # deleting labels from samples- train set
    x_test = landsat_test_normalized[:, :-1]  # deleting labels from samples- test set
    num_of_samples = 1000
    x_train = x_train[:num_of_samples]
    x_test = x_test[:num_of_samples]
    y_train = y_train[:num_of_samples]
    y_test = y_test[:num_of_samples]
    label_encoder = preprocessing.LabelEncoder()
    y = np.concatenate((y_train, y_test))
    print(y.shape)
    label_encoder.fit(y)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    ##y_train = landsat_train_normalized[:, -1]  # labels of samples- train set
    ##y_test = landsat_test_normalized[:, -1]  # labels of samples- test set
    # analyze_all_models(x_train, x_test, y_train, y_test, "landsat")
    clf = LogisticRegression(random_state=42, max_iter=150)
    y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy, mistake_prob = run_inference_attacks(
        x_train,
        x_test,
        y_train,
        y_test,
        clf)
    Print("y_pred_train shape", y_pred_train.shape)
    Print("y_pred_test shape", y_pred_test.shape)
    K = [2, 3, 5, 11]
    P = [2.0, 1.0]
    for p in P:
        for k in K:
            Print(f"p={p}, k={k}, MI train: ", mi.mi_Kraskov_HnM(x_train, y_pred_train, k=k, p_x=p, p_y=p))
            Print(f"p={p}, k={k}, MI test: ", mi.mi_Kraskov_HnM(x_test, y_pred_test, k=k, p_x=p, p_y=p))
            Print(f"p={p}, k={k},entropy orig: ", mi.entropy(x_train, k=k))
            Print(f"p={p}, k={k},entropy p norm: ", mi.entropy(x_train, k=k, p=p))
            Print(f"p={p}, k={k},entropy p norm with correction: ", mi.entropy_with_correction(x_train, k=k, p=p))


def test_loan_with_miltiple_models():
    Print("testing with loan data...")
    loan = np.loadtxt('loan_train_from_abigail.csv', delimiter=',')
    loan_normalized = preprocessing.MinMaxScaler().fit_transform(loan)
    loan_normalized = loan_normalized[:2000]
    x = loan_normalized[:, :-1]  # deleting labels from samples
    y = loan_normalized[:, -1]  # labels of samples
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
    analyze_all_models(x_train, x_test, y_train, y_test, 'loan')


def unpickle_and_plot_results2():
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
            LB_p = LB_.loc[LB_["p"] == p, "LB on prob.., p with correction"]
            acc_bb = acc_.loc[acc_["p"] == p, "attack acc bb"]
            acc_ldb = acc_.loc[acc_["p"] == p, "attack acc ldb"]
            mi_p = mi_.loc[mi_["p"] == p, "mi"]
            LB_p = LB_p.tolist()
            acc_bb = acc_bb.tolist()
            acc_ldb = acc_ldb.tolist()
            mi_p = mi_p.tolist()
            assert len(mi_p) == len(acc_ldb) == len(acc_bb) == len(LB_p)

            plt.figure()
            for i in range(len(LB_p)):
                j = i % len(markers)
                l = int(i /(len(LB_p)/len(datasets)))
                plt.scatter(LB_p[i], acc_bb[i], label=f"bb, {datasets[l]} {models[j]}", color=colors[l], marker=markers[j])
            plt.title(f"LB VS. bb attack accuracy- p={p}, k={k}")
            plt.xlabel("LB")
            plt.ylabel("attack accuracy")
            plt.legend()
            plt.savefig(f'LB VS bb attack acc p={p} k={k}.png')
            plt.close()

            plt.figure()
            for i in range(len(LB_p)):
                j = i % len(markers)
                l = int(i /(len(LB_p)/len(datasets)))
                plt.scatter(LB_p[i], acc_ldb[i], label=f"ldb, {datasets[l]} {models[j]}", color=colors[l], marker=markers[j])
            plt.title(f"LB VS. ldb attack accuracy- p={p}, k={k}")
            plt.xlabel("LB")
            plt.ylabel("attack accuracy")
            plt.legend()
            plt.savefig(f'LB VS ldb attack acc p={p} k={k}.png')
            plt.close()

            plt.figure()
            for i in range(len(mi_p)):
                j = i % len(markers)
                l = int(i / (len(mi_p) / len(datasets)))
                plt.scatter(mi_p[i], acc_bb[i], label=f"bb, {datasets[l]} {models[j]}", color=colors[l],marker=markers[j])
            plt.title(f"MI VS. bb attack accuracy- p={p}, k={k}")
            plt.xlabel("MI")
            plt.ylabel("attack accuracy")
            plt.legend()
            plt.savefig(f'MI VS bb attack acc p={p} k={k}.png')
            plt.close()

            plt.figure()
            for i in range(len(mi_p)):
                j = i % len(markers)
                l = int(i / (len(mi_p) / len(datasets)))
                plt.scatter(mi_p[i], acc_ldb[i], label=f"ldb, {datasets[l]} {models[j]}", color=colors[l],
                            marker=markers[j])
            plt.title(f"MI VS. ldb attack accuracy- p={p}, k={k}")
            plt.xlabel("MI")
            plt.ylabel("attack accuracy")
            plt.legend()
            plt.savefig(f'MI VS ldb attack acc p={p} k={k}.png')
            plt.close()


def unpickle_and_plot_results():
    data = []
    with open("picklefile", 'rb') as pf:
        try:
            while True:
                data.append(pickle.load(pf))
        except EOFError:
            pass
    df = pd.DataFrame(data)
    pd.DataFrame.to_csv(df, "res.csv")
    # df = df.head(n=87)
    # print(len(df))
    K = df["k"].unique()
    P = df["p"].unique()
    datasets = df["data set"].unique()
    colors = ['r', 'g', 'b']
    markers = [".", "v", "p", "+"]
    assert len(P) == len(colors)
    for ds in datasets:
        mis = df.loc[df["data set"] == ds]
        attack_acc = df.loc[df["data set"] == ds]
        model = df.loc[df["data set"] == ds]
        for k in K:
            mis_k = mis.loc[mis["k"] == k, "mi"]
            attack_acc_bb_k = attack_acc.loc[attack_acc["k"] == k, "attack acc bb"]
            attack_acc_ldb_k = attack_acc.loc[attack_acc["k"] == k, "attack acc ldb"]
            model_k = model.loc[model["k"] == k, "model"]
            assert len(attack_acc_bb_k) == len(mis_k) == len(model_k)
            plt.figure()
            for i in range(len(P)):
                mis_p = mis_k[i::len(P)]
                attack_acc_bb_p = attack_acc_bb_k[i::len(P)]
                attack_acc_ldb_p = attack_acc_ldb_k[i::len(P)]
                model_p = model_k[i::len(P)]
                mis_p = mis_p.tolist()
                attack_acc_bb_p = attack_acc_bb_p.tolist()
                attack_acc_ldb_p = attack_acc_ldb_p.tolist()
                model_p.values.tolist()
                plt.scatter(mis_p[0], attack_acc_bb_p[0], label=f"bb, p={P[i]} lr", color=colors[i], marker=markers[0])
                plt.scatter(mis_p[1], attack_acc_bb_p[1], label=f"bb, p={P[i]} dt", color=colors[i], marker=markers[1])
                plt.scatter(mis_p[2], attack_acc_bb_p[2], label=f"bb, p={P[i]} mlp", color=colors[i], marker=markers[2])
                plt.scatter(mis_p[3], attack_acc_bb_p[3], label=f"bb, p={P[i]} rf", color=colors[i], marker=markers[3])
                plt.scatter(mis_p[0], attack_acc_ldb_p[0], label=f"ldb, p={P[i]} lr", color=colors[i],
                            marker=markers[0])
                plt.scatter(mis_p[1], attack_acc_ldb_p[1], label=f"ldb, p={P[i]} dt", color=colors[i],
                            marker=markers[1])
                plt.scatter(mis_p[2], attack_acc_ldb_p[2], label=f"ldb, p={P[i]} mlp", color=colors[i],
                            marker=markers[2])
                plt.scatter(mis_p[3], attack_acc_ldb_p[3], label=f"ldb, p={P[i]} rf", color=colors[i],
                            marker=markers[3])
            plt.title(f"MI(D,Y) VS. bb attack accuracy- {ds}, k={k}")
            plt.xlabel("MI")
            plt.ylabel("attack accuracy")
            plt.legend()
            # plt.show()
            plt.savefig(f'MI(D,Y) VS attack acc {ds} k={k}.png')

        # print (f"K IS {k}")
        # print(mis)


def main():
    Print("----- STARTING TESTS! ------")
    test_german_with_multiple_models()
    test_landsat_with_multiple_models()
    test_loan_with_miltiple_models()
    #unpickle_and_plot_results()
    unpickle_and_plot_results2()
    # test_landsat_manually()

    Print("----- DONE :) -----")


def Print(string, *args):
    print(string, *args)
    print(string, *args, file=output)
    # output.flush()


if __name__ == "__main__":
    output = open("log.txt", "a", 1)
    main()
    output.close()

'''


###linear regression model from KAGGLE https://www.kaggle.com/kanncaa1/logistic-regression-implementation/data


#%% import dataset
#data = pd.read_csv("../input/data.csv")
#data.drop(['Unnamed: 32',"id"], axis=1, inplace=True)
#data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
#y = data.diagnosis.values
#x_data = data.drop(['diagnosis'], axis=1)

y = german[:, -1]



# %% normalization
x = (x_data -np.min(x_data))/(np.max(x_data)-np.min(x_data)).values



# %%train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

Print("x train: ",x_train.shape)
Print("x test: ",x_test.shape)
Print("y train: ",y_train.shape)
Print("y test: ",y_test.shape)



# %%initialize
# lets initialize parameters
# So what we need is dimension 4096 that is number of pixels as a parameter for our initialize method(def)
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b


# %% sigmoid
# calculation of z
# z = np.dot(w.T,x_train)+b
def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head
# y_head = sigmoid(5)




#%% forward and backward
# In backward propagation we will use y_head that found in forward progation
# Therefore instead of writing backward propagation method, lets combine forward propagation and backward propagation
def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost,gradients



#%%# Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            Print ("Cost after iteration %i: %f" %(i, cost))
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    #plt.plot(index,cost_list2)
    #plt.xticks(index,rotation='vertical')
    #plt.xlabel("Number of Iterarion")
    #plt.ylabel("Cost")
    #plt.show()
    return parameters, gradients, cost_list

#%%  # prediction
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction
# predict(parameters["weight"],parameters["bias"],x_test)


# %%
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    # initialize
    dimension = x_train.shape[0]  # that is 4096
    w, b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)

    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)

    # Print train/test Errors
    Print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    Print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))


logistic_regression(x_train, y_train, x_test, y_test, learning_rate=1, num_iterations=100)



# sklearn
from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)
Print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
Print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))



'''
