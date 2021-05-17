from art.metrics.privacy.MIestimator import MI as mi
from art.attacks.inference.membership_inference.black_box import MembershipInferenceBlackBox
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from matplotlib import pyplot as plt
from math import comb, log


## p_aplpha >= lower bound. p_alpha is the probability of MIA model making more then alpha prediction errors
def calc_LB_prob_MIA_errors(training_data_D, output_Y, size_of_data_set, alpha):
    assert alpha > 0 and isinstance(alpha, int)
    mi_D_Y = mi.mi_Kraskov_HnM(training_data_D, output_Y)
    entropy_D = mi.entropy(training_data_D)
    const = log(sum([comb(size_of_data_set, i) for i in range(alpha + 1)]))
    return (entropy_D - mi_D_Y - 1 - const) / (size_of_data_set - const)


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


german = np.loadtxt('/home/hilla/Hilla/tehnion/IBM project/dataset statlog/german.data-numeric.csv', delimiter=',')
min_max_scaler = preprocessing.MinMaxScaler()
german_normalized = min_max_scaler.fit_transform(german)
x = german_normalized[:, :-1]  # deleting labels from samples
y = german_normalized[:, -1]  # labels of samples
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
print("train data shape: ", x_train.shape)
print("test data shape: ", x_test.shape)
print("train labels shape: ", y_train.shape)
print("test labels shape: ", y_test.shape)

logreg = LogisticRegression(random_state=42, max_iter=150)
logreg.fit(x_train, y_train)

y_pred_train = logreg.predict(x_train)
y_pred_test = logreg.predict(x_test)
y_pred_train = np.array([y_pred_train]).T
y_pred_test = np.array([y_pred_test]).T

classifier = ScikitlearnClassifier(logreg)
attack = MembershipInferenceBlackBox(classifier)
## Naming of vars explained:
## x/ y always indicates data and labels accordingly.
## First train\ test indicates if the model w`as trained on this data or not.
## Second train\ test indicates if the bb attack was trained on this data or not. bb atack trains on both data that was
# used to train the model AND data that was NOT used to train the model
x_train_attack_train, x_train_attack_test, y_train_attack_train, y_train_attack_test = train_test_split(x_train,
                                                                                                        y_train,
                                                                                                        test_size=0.5,
                                                                                                        random_state=42)
x_test_attack_train, x_test_attack_test, y_test_attack_train, y_test_attack_test = train_test_split(x_test, y_test,
                                                                                                    test_size=0.5,
                                                                                                    random_state=42)
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

infered_membership = attack.infer(x_attack_test, y_attack_test)
print(infered_membership.shape)
TN, FP, FN, TP = confusion_matrix(true_membership, infered_membership).ravel()
print("TN ", TN)
print("FP ", FP)
print("FN ", FN)
print("TP ", TP)
accuracy = (TP + TN) / (TN + FP + FN + TP)
print("accuracy ", accuracy)

K = [*range(1, 16)]
# K = [2, 3, 5, 10]
P = [float('inf'), 2.0, 1.0]

mi_train = []
mi_test = []
for p in P:
    mi_train.append([])
    mi_test.append([])
    for k in K:
        mi_train[-1].append(mi.mi_Kraskov_HnM(x_train, y_pred_train, k=k, p_x=p))
        mi_test[-1].append(mi.mi_Kraskov_HnM(x_test, y_pred_test, k=k, p_x=p))
    plot_mi(K, mi_train[-1], title='train', p=p)
    plot_mi(K, mi_test[-1], title='test', p=p)

# np.save("mi_train", mi_train)
# np.save("mi_test", mi_test)
# mi_train = np.load("mi_train.npy")
# mi_test = np.load("mi_test.npy")

xticks = np.arange(len(K))
K_strings = [f'k = {k}' for k in K]
bar_plot_mi(xticks, y=mi_train, width=0.25, labels=K_strings, legends=P, n_bars=len(P), title='train')
bar_plot_mi(xticks, y=mi_test, width=0.25, labels=K_strings, legends=P, n_bars=len(P), title='test')

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
