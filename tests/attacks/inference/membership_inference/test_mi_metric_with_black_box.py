from art.metrics.privacy.MIestimator import MI as mi
from art.attacks.inference.membership_inference.black_box import MembershipInferenceBlackBox
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
from matplotlib import pyplot as plt
from math import comb, log


## p_aplpha >= lower bound. p_alpha is the probability of MIA model making more then alpha prediction errors
def calc_LB_prob_MIA_errors(training_data_D, output_Y, size_of_data_set, Alphas, k, p):
    mi_D_Y = mi.mi_Kraskov_HnM(training_data_D, output_Y, k=k, p_x=p)
    print(f'MI(D,Y)={mi_D_Y}, k={k}, p={p}')
    entropy_D = mi.entropy(training_data_D, k=k, p=p)
    p_alphas = []
    for alpha in Alphas:
        const = log(sum([comb(size_of_data_set, i) for i in range(alpha + 1)]))
        p_alphas.append((entropy_D - mi_D_Y - 1 - const) / (size_of_data_set - const))
    return p_alphas


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
    return sum(
        [comb(size_of_data_set, i) * (mistake_prob ** i) * ((1 - mistake_prob) ** (size_of_data_set - i)) for i in
         range(alpha, size_of_data_set + 1)])


# alpha is the number of mistakes that attack makes
def plot_probabilities_as_func_of_alpha(LBs, p_alphas, Alphas, K, p, model_name, data_set_name):
    plt.figure()
    plt.plot(Alphas, p_alphas, label='empirical p_alpha')
    for i in range(len(K)):
        plt.plot(Alphas, LBs[i], label=f'LB with k={K[i]}')
    plt.title(
        f"probability of MIA attack making alpha prediction errors: empirical estimation vs. lower bound with mi - {model_name}, {data_set_name}, p_norm={p}")
    plt.xlabel("alpha (number of prediction errors of MIA attack)")
    plt.ylabel("probability")
    plt.legend()
    plt.savefig(f'prob(alpha), p={p}, {model_name}, {data_set_name}.png')


def run_bb_attack(x_train, x_test, y_train, y_test, model):
    model.fit(x_train, y_train)
    y_pred_train = model.predict_proba(x_train)[:, -1]
    y_pred_test = model.predict_proba(x_test)[:, -1]
    y_pred_train = np.array([y_pred_train]).T
    y_pred_test = np.array([y_pred_test]).T
    accuracy_train = accuracy_score(y_train, model.predict(x_train))
    accuracy_test = accuracy_score(y_test, model.predict(x_test))
    print("accuracy of model's predictions on train set: ", accuracy_train)
    print("accuracy of model's predictions on test set: ", accuracy_test)

    ## Naming of vars explained:
    ## x/ y always indicates data and labels accordingly.
    ## First train\ test indicates if the model was trained on this data or not.
    ## Second train\ test indicates if the bb attack was trained on this data or not. bb atack trains on both data that was
    # used to train the model AND data that was NOT used to train the model
    x_train_attack_train, x_train_attack_test, y_train_attack_train, y_train_attack_test = train_test_split(x_train,
                                                                                                            y_train,
                                                                                                            test_size=0.5,
                                                                                                            random_state=42)
    x_test_attack_train, x_test_attack_test, y_test_attack_train, y_test_attack_test = train_test_split(x_test, y_test,
                                                                                                        test_size=0.5,
                                                                                                        random_state=42)
    classifier = ScikitlearnClassifier(model)
    attack = MembershipInferenceBlackBox(classifier)
    attack.fit(x_train_attack_train, y_train_attack_train, x_test_attack_train, y_test_attack_train)

    x_attack_test = np.concatenate((x_train_attack_test, x_test_attack_test))
    y_attack_test = np.concatenate((y_train_attack_test, y_test_attack_test))
    # print(x_attack_test.shape)
    # print(y_attack_test.shape)
    assert x_attack_test.shape[0] == y_attack_test.shape[0]
    ones = np.ones(x_train_attack_test.shape[0])
    zeros = np.zeros(x_test_attack_test.shape[0])
    true_membership = np.concatenate((ones, zeros))
    # print(true_membership.shape)
    # print(y_attack_test.shape)

    inferred_membership = attack.infer(x_attack_test, y_attack_test)
    # print(infered_membership.shape)
    TN, FP, FN, TP = confusion_matrix(true_membership, inferred_membership).ravel()
    print("confusion mat of inference attack:")
    print("TN ", TN)
    print("FP ", FP)
    print("FN ", FN)
    print("TP ", TP)
    attack_accuracy = (TP + TN) / (TN + FP + FN + TP)
    # attack_accuracy = accuracy_score(true_membership, inferred_membership)
    assert attack_accuracy == accuracy_score(true_membership, inferred_membership)
    mistake_prob = (FP + FN) / (TN + FP + FN + TP)
    print("black box attack accuracy ", attack_accuracy)
    return y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy, mistake_prob


def analyze_logistic_regression(x_train, x_test, y_train, y_test, data_set_name):
    print(f'    ##running black box attack - Logistic Regression')
    clf = LogisticRegression(random_state=42, max_iter=150)
    y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy, mistake_prob = run_bb_attack(x_train,
                                                                                                            x_test,
                                                                                                            y_train,
                                                                                                            y_test,
                                                                                                            clf)
    analyze(x_train, x_test, y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy, mistake_prob,
            "Logistic Regression", data_set_name)


def analyze_decision_tree(x_train, x_test, y_train, y_test, data_set_name):
    print(f'    ##running black box attack - Decision Tree')
    clf = DecisionTreeClassifier(random_state=0)
    y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy, mistake_prob = run_bb_attack(x_train,
                                                                                                            x_test,
                                                                                                            y_train,
                                                                                                            y_test,
                                                                                                            clf)
    analyze(x_train, x_test, y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy, mistake_prob,
            "Decision Tree", data_set_name)


def analyze_sklearn_MLPClassifier(x_train, x_test, y_train, y_test, data_set_name):
    print(f'    ##running black box attack - MLP')
    clf = MLPClassifier(random_state=0)
    y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy, mistake_prob = run_bb_attack(x_train,
                                                                                                            x_test,
                                                                                                            y_train,
                                                                                                            y_test,
                                                                                                            clf)
    analyze(x_train, x_test, y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy, mistake_prob,
            "MLP", data_set_name)


def analyze_RandomForestClassifier(x_train, x_test, y_train, y_test, data_set_name):
    print(f'    ##running black box attack - Random Forest')
    clf = RandomForestClassifier(random_state=0)
    y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy, mistake_prob = run_bb_attack(x_train,
                                                                                                            x_test,
                                                                                                            y_train,
                                                                                                            y_test,
                                                                                                            clf)
    analyze(x_train, x_test, y_pred_train, y_pred_test, accuracy_train, accuracy_test, attack_accuracy,
            mistake_prob, "Random Forest", data_set_name)


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
    LB = []
    p_alphas = [calc_prob_of_alpha_mistakes(mistake_prob, alpha, size_of_data_set) for alpha in Alphas]
    for p in P:
        mi_train.append([])
        mi_test.append([])
        LB.append([])
        for k in K:
            mi_train[-1].append(mi.mi_Kraskov_HnM(x_train, y_pred_train, k=k, p_x=p))
            mi_test[-1].append(mi.mi_Kraskov_HnM(x_test, y_pred_test, k=k, p_x=p))
            LB[-1].append(calc_LB_prob_MIA_errors(x_train, y_pred_train, size_of_data_set, Alphas, k=k, p=p))
        # plot_mi(K, mi_train[-1], title='train', p=p)
        # plot_mi(K, mi_test[-1], title='test', p=p)
        plot_probabilities_as_func_of_alpha(LB[-1], p_alphas, Alphas, K, p, model_name, data_set_name)

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
    print("train data shape: ", x_train.shape)
    print("test data shape: ", x_test.shape)
    print("train labels shape: ", y_train.shape)
    print("test labels shape: ", y_test.shape)
    analyze_logistic_regression(x_train, x_test, y_train, y_test, data_set_name)
    analyze_decision_tree(x_train, x_test, y_train, y_test, data_set_name)
    analyze_sklearn_MLPClassifier(x_train, x_test, y_train, y_test, data_set_name)
    analyze_RandomForestClassifier(x_train, x_test, y_train, y_test, data_set_name)


def test_german_with_multiple_models():
    print("testing with german data...")
    german = np.loadtxt('german.data-numeric.csv', delimiter=',')
    german_normalized = preprocessing.MinMaxScaler().fit_transform(german)
    x = german_normalized[:, :-1]  # deleting labels from samples
    y = german_normalized[:, -1]  # labels of samples
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
    analyze_all_models(x_train, x_test, y_train, y_test, "german")


def test_landsat_with_multiple_models():
    print("testing with landsat data...")
    landsat_train = np.loadtxt('sat.trn', delimiter=',')
    landsat_test = np.loadtxt('sat.tst', delimiter=',')
    landsat_train_normalized = preprocessing.MinMaxScaler().fit_transform(landsat_train)
    landsat_test_normalized = preprocessing.MinMaxScaler().fit_transform(landsat_test)
    x_train = landsat_train_normalized[:, :-1]  # deleting labels from samples- train set
    x_test = landsat_test_normalized[:, :-1]  # deleting labels from samples- test set
    y_train = landsat_train_normalized[:, -1]  # labels of samples- train set
    y_test = landsat_test_normalized[:, -1]  # labels of samples- test set
    analyze_all_models(x_train, x_test, y_train, y_test, "landsat")


def main():
    test_german_with_multiple_models()
    test_landsat_with_multiple_models()


if __name__ == "__main__":
    main()

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

print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)



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
            print ("Cost after iteration %i: %f" %(i, cost))
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
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))


logistic_regression(x_train, y_train, x_test, y_test, learning_rate=1, num_iterations=100)



# sklearn
from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)
print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))



'''
