import numpy as np
import sys

class NeuralNetMLP():
    """
    parameters
    n_hidden : int
        default 30
    l2 : float
        default 0.
        L2 regularization parameter lambda
    epochs : int
        default 100
    eta : float
        default 0.001
    shuffle : bool
        default True
        True の場合循環を避けるためエポックごとに訓練データをシャッフル
    minibatch_size : int
        default 1
    seed : int
        default None
        重みとシャッフルを初期化するための乱数シード

    attributes
    eval_ : dict
        訓練のエポックごとにコスト、訓練の正解率、検証の正解率を収集する辞書
    """

    def __init__(self, n_hidden=30, l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """
        ラベルを one-hot 表現にエンコードする
        y : array, shape = [n_examples]

        onehot : array, shape = (n_examples, n_labels)
        """

        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        
        return onehot.T
    
    def _sigmoid(self, z):
        """sigmoid を計算. 活性化関数で用いる"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def _forward(self, X):
        """forward propagationのステップの計算"""
        # step 1 : 隠れ層の総入力
        # [n_examples, n_features] dot [n_features, n_hidden] 
        # -> [n_examples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h

        # step 2 : 隠れ層の活性化関数
        a_h = self._sigmoid(z_h)

        # step 3 : 出力層の総入力
        # [n_examples, n_hidden] dot [n_hidden, n_classlabels]
        # [n_examples, n_classlabels]
        z_out = np.dot(a_h, self.w_out) + self.b_out

        # step 4 : 出力層の活性化関数
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out
    
    def _compute_cost(self, y_enc, output):
        """loss を計算
        parameters
        y_enc : array, shape = (n_examples, n_labels)
            one-hot encoding されたクラスラベル
        output : array, shape = [n_examples, n_output_units]
            出力層の活性化関数

        return
        cost : float
        """

        L2_term = (self.l2 * (np.sum(self.w_h ** 2.) +
                              np.sum(self.w_out ** 2.)))
        term1 = -y_enc * (np.log(output + 1e-5))
        term2 = (1. - y_enc) * np.log(1. - output + 1e-5)
        cost = np.sum(term1 - term2) + L2_term

        return cost
    
    def predict(self, X):
        """ クラスラベルの予測
        parameters
        X : array, shape = [n_examples, n_features]
            元の特徴量が設定された入力層
        return
        y_pred : array, shape = [n_examples]
        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)

        return y_pred
    
    def fit(self, X_train, y_train, X_valid, y_valid):
        """
        parameters
        X_train : array, shape = [n_examples, n_features]
        y_train : array, shape = [n_examples]
        X_valid : array, shape = [n_examples, n_features]
        y_valid : array, shape = [n_examples]
        return
        self
        """

        n_output = np.unique(y_train).shape[0]
        n_features = X_train.shape[1]
        
        # weight initialize

        # input layer -> hidden layer
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))
        
        # hidden layer -> output layer
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                       size=(self.n_hidden, n_output))
        
        # settings
        epoch_strlen = len(str(self.epochs))
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}

        y_train_enc = self._onehot(y_train, n_output)

        for i in range(self.epochs):
            # minibatch iteration
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)
            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1,
                                   self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                # back propagation
                # [n_examples, n_classlabels]
                delta_out= a_out - y_train_enc[batch_idx]

                # [n_examples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # [n_examples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_examples, n_hidden]
                delta_h = (np.dot(delta_out, self.w_out.T) * sigmoid_derivative_h)

                # [n_features, n_examples] dot [n_examples, n_hidden]
                # -> [n_features, n_hidden]
                grad_w_h = np.dot(X_train[batch_idx].T, delta_h)
                grad_b_h = np.sum(delta_h, axis=0)

                # [n_hidden, n_examples] dot [n_examples, n_classlabels]
                # -> [n_hidden, n_classlabels]
                grad_w_out = np.dot(a_h.T, delta_out)
                grad_b_out = np.sum(delta_out, axis=0)

                # 正則化と重みの更新
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out
            
            # evaluations
            # evaluation by iteration
            z_h, a_h, z_out, a_out = self._forward(X_train)
            cost = self._compute_cost(y_enc=y_train_enc, output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float64) / X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float64) / X_valid.shape[0])

            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              train_acc*100, valid_acc*100))
            
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        # method chain できるようにしておく
        return self

