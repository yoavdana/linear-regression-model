import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg as lin


def load_data(filename):
    '''
    loadin the data,and filter wrong and unresoable data,split the data to X (design matrix) an y (response vector)
    :param filename:
    :return:X,y
    '''
    data=pd.read_csv(filename).dropna().drop_duplicates()
    data["zipcode"]=data["zipcode"].astype(int)

    for columb in ["id", "lat", "long", "date"]:
        data = data.drop(columb, 1)
    for columb in ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built", "sqft_living15", "sqft_lot15"]:
        data=data[data[columb]>0]
    for columb in ["bathrooms", "floors", "sqft_basement", "yr_renovated"]:
        data = data[data[columb] >= 0]
    data = data[data['waterfront'].isin([0,1])& data['view'].isin(range(5))&
                data['condition'].isin(range(1,6))& data['grade'].isin(range(1,
        15))]
    data["recently_renovated"] = np.where(
        data["yr_renovated"] >= np.percentile(data.yr_renovated.unique(),
                                              70), 1,
        0)
    data = data.drop("yr_renovated", 1)
    data = data[data["bedrooms"] < 20]
    data = data[data["sqft_lot"] < 1250000]
    data = data[data["sqft_lot15"] < 500000]

    data["decade_built"] = (data["yr_built"] / 10).astype(int)
    data = data.drop("yr_built", 1)

    data = pd.get_dummies(data, prefix='zipcode_', columns=['zipcode'])
    data = pd.get_dummies(data, prefix='decade_built_', columns=[
        'decade_built'])
    return data.drop("price", 1), data.price


def plot_singular_points(singular_values):
    '''
    plot all sigular values of design matrix X
    :param singular_values:
    :return:
    '''
    plt.figure()
    sing_vals = np.arange(singular_values.shape[0]) + 1
    plt.plot(sing_vals, singular_values, 'ro-', linewidth=1)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Singular Values')
    plt.show()


def fit_linear_regression(X,y):
    '''

    :param X:np array mXd, of the data
    :param y:response vector
    :return: w,wheights vector of the linear regression estimator
    singular values of X
    '''

    return  lin.pinv(X) @ y,lin.svd(X,full_matrices=False)[1]


def predict(X,w):
    '''
    make the prediction of the linear regression model
    :param X:
    :param w:
    :return: prediction vector y
    '''
    return X @ w

def mse(y_hat,y):
    '''
    calculate the MSE
    :param y_hat:
    :param y:
    :return: MSE
    '''
    est=(y-y_hat)**2
    return np.mean(est)

def putting_it_all_1(filename):
    '''
    plot the sigular values of the design matrix of the model
    :param filename:
    :return:
    '''
    X,y=load_data(filename)
    w,singular_vals=fit_linear_regression(X,y)
    plot_singular_points(singular_vals)

def putting_it_all2(filename):
    '''
    show the MSE of the full model as a function of size of the training set
    :param filename:
    :return:
    '''
    X, y = load_data(filename)
    X=pd.DataFrame.to_numpy(X)
    y = pd.DataFrame.to_numpy(y)
    train_size = int(0.75 * len(y))
    X_train = X[:train_size, :]
    y_train = y[:train_size]
    X_test = X[train_size:, :]
    y_test = y[train_size:]
    mse_vec = np.zeros(100)
    for i in range(1, 101):
        n = int(train_size * (i / 100))
        w, s = fit_linear_regression(X_train[:n + 1, :], y_train[:n + 1])
        y_test_h = predict(X_test, w)
        mse_vec[i - 1] = mse(y_test_h, y_test)
    plt.plot(mse_vec)
    plt.title("MSE as a function of p%")
    plt.xlabel("Percent")
    plt.ylabel("MSE")
    plt.show()

def feature_evaluation(csv_file):
    '''
    sohw corelation betwwen features
    :param csv_file:
    :return:
    '''
    df = pd.read_csv(csv_file)
    df = df.dropna()
    X = df.drop('price', axis=1)
    X = X.drop('date', axis=1)
    col = np.array(X.columns)
    X = X.to_numpy()
    y = df['price'].to_numpy()
    for i in range(X.shape[1]):
        cov_mat = np.cov(y, X[:, i])
        corr = cov_mat[0, 1] / ((cov_mat[0, 0] ** 0.5) * (cov_mat[1,
                                                                  1] ** 0.5))
        plt.scatter(X[:, i], y)
        plt.title(col[i] + "\n Pearson correlation:" + str(corr))
        plt.xlabel(col[i])
        plt.ylabel("Price")
        plt.show()

