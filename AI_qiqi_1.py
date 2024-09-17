import math
import numpy as np


# sigmoid函数
def basic_sigmoid(X):
    s = 1.0 / (1 - math.exp(-X))
    return s


# 初始化网络参数
def initialize_with_zeros(shape):
    """ 创建一个形状为（shape，1）的 w 参数和 b=0
            return：w，b
    """
    w = np.zeros((shape, 1))
    b = 0
    # 断言检查
    assert (w.shape == (shape, 1))
    assert (isinstance(b, float)) or isinstance(b, int)
    # 确保 b 是一个浮点数或整数。如果 b 的类型不符合要求，程序将抛出 AssertionError

    return w, b


# 前向传播和反向传播
def propagate(w, b, X, Y):
    """
    :param w: 权重，numpy 数组，形状为 (num_features, 1)
    :param b: 偏置，一个标量
    :param X: 数据集，形状为 (num_features, num_examples)
    :param Y: 标签向量，形状为 (1, num_examples)
    :return: dw -- 权重的梯度，形状为 (num_features, 1)
             db -- 偏置的梯度，一个标量
             cost -- 代价函数的值
    """
    num_examples = X.shape[1]
    """  
    X.shape[0] 表示数组的第一维（即行数），即特征的数量。
    X.shape[1] 表示数组的第二维（即列数），即样本的数量。
    """
    # 前向传播
    # w.T.shape=(1,num_features)  X.shape=(num_features,num_examples)
    A = basic_sigmoid(np.dot(w.T, X) + b)
    # 计算损失
    cost = -1 / num_examples * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # 反向传播
    dz = A - Y
    dw = 1 / num_examples * np.dot(X, dz.T)
    db = 1 / num_examples * np.sum(dz)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)  # 去除了 cost 中的单维度条目，使其成为一个标量
    assert (cost.shape == ())  # 确保 cost 的形状为空元组，即 cost 是一个标量
    grads = {"dw": dw,
             "db": db}  # grads 是一个字典，包含了权重和偏置的梯度
    return grads, cost


# 优化过程
def optimize(w, b, X, Y, num_iterations, learning_rate):
    """

    :param w: 权重，numpy 数组，形状为 (num_features, 1)
    :param b: 偏置，一个标量
    :param X: 数据集，形状为 (num_features, num_examples)
    :param Y: 标签向量，形状为 (1, num_examples)
    :param num_iterations: 总迭代次数
    :param learning_rate: 学习率
    :return: params: 更新后的参数字典
             grads: 权重和偏置的梯度字典
             costs: 损失结果
    """
    costs = []
    for i in range(num_iterations):
        # 梯度更新计算函数 开始
        grads, cost = propagate(w, b, X, Y)

        # 取出梯度
        dw = grads['dw']
        db = grads['db']
        # 按照梯度下降公式更新
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # 结束

        if i % 100 == 0:  # 每一百次记录一下损失结果，便于观察
            costs.append(cost)
        if i % 100 == 0:
            print("损失结果 %i:  %f" % (i, cost))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs


# 预测函数
def predict(w, b, X):
    """
    :param X: 要预测的数据，形状为 (num_features, num_examples)
    利用训练好的参数预测
    :return: Y_prediction 预测结果，形状为 (1, num_examples)
    """
    num_examples = X.shape[1]
    Y_prediction = np.zeros((1, num_examples))  # 用于存放预测结果
    w = w.reshape(X.shape[0], 1)  # 确保 w 的形状与输入数据 X 的第一维大小一致，便于后续的矩阵运算

    # 计算预测结果 开始
    Z = np.dot(w.T, X) + b
    A = basic_sigmoid(Z)
    # 结束
    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    assert (Y_prediction.shape == (1, num_examples))
    return Y_prediction


# 整体逻辑实现
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5):
    # 初始化参数
    w, b = initialize_with_zeros(X_train.shape[0])
    # 梯度下降 params:更新后的网络参数，grads:最后一次梯度，costs:记录的更新损失列表
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    # 获取训练的参数
    w = params['w']
    b = params['b']

    # 预测结果
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # 打印准确率
    print("训练集准确率：{}".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("测试集集准确率：{}".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {
        "costs": costs,
        "Y_prediction_train": Y_prediction_train,
        "Y_prediction_test": Y_prediction_test,
        "w": w,
        "b": b,
        "learning_rete": learning_rate,
        "num_iterations": num_iterations
    }
    return d
