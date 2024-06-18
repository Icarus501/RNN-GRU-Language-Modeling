# coding: utf-8
from rnnmath import *
from model import Model, is_param, is_delta


class RNN(Model):
    '''
    This class implements Recurrent Neural Networks.

    You should implement code in the following functions:
        predict				->	predict an output sequence for a given input sequence
        acc_deltas			->	accumulate update weights for the RNNs weight matrices, standard Back Propagation
        acc_deltas_bptt		->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time
        acc_deltas_np		->	accumulate update weights for the RNNs weight matrices, standard Back Propagation -- for number predictions
        acc_deltas_bptt_np	->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time -- for number predictions

    Do NOT modify any other methods!
    Do NOT change any method signatures!
    '''

    def __init__(self, vocab_size, hidden_dims, out_vocab_size):
        '''
        initialize the RNN with random weight matrices.

        DO NOT CHANGE THIS

        vocab_size		size of vocabulary that is being used
        hidden_dims		number of hidden units
        out_vocab_size	size of the output vocabulary
        '''

        super().__init__(vocab_size, hidden_dims, out_vocab_size)

        # matrices V (input -> hidden), W (hidden -> output), U (hidden -> hidden)
        with is_param():
            self.U = np.random.randn(self.hidden_dims, self.hidden_dims) * np.sqrt(0.1)
            self.V = np.random.randn(self.hidden_dims, self.vocab_size) * np.sqrt(0.1)
            self.W = np.random.randn(self.out_vocab_size, self.hidden_dims) * np.sqrt(0.1)

        # matrices to accumulate weight updates
        with is_delta():
            self.deltaU = np.zeros_like(self.U)
            self.deltaV = np.zeros_like(self.V)
            self.deltaW = np.zeros_like(self.W)

    def predict(self, x):
        '''
        predict an output sequence y for a given input sequence x

        x	list of words, as indices, e.g.: [0, 4, 2]

        returns	y,s
        y	matrix of probability vectors for each input word
        s	matrix of hidden layers for each input word

        '''

        # matrix s for hidden states, y for output states, given input x.
        # rows correspond to times t, i.e., input words
        # s has one more row, since we need to look back even at time 0 (s(t=0-1) will just be [0. 0. ....] )

        s = np.zeros((len(x) + 1, self.hidden_dims))
        y = np.zeros((len(x), self.out_vocab_size))

        for t in range(len(x)):
            ##########################
            encoded_x = make_onehot(x[t], self.out_vocab_size)
            net_in = np.matmul(self.V, encoded_x) + np.matmul(self.U, s[t - 1])  # S[t-1] ***
            s[t] = sigmoid(net_in)
            net_out = np.matmul(self.W, s[t])
            y[t] = softmax(net_out)
        ##########################
        return y, s

    def acc_deltas(self, x, d, y, s):
        '''
        accumulate updates for V, W, U
        standard back propagation

        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time

        x	list of words, as indices, e.g.: [0, 4, 2]
        d	list of words, as indices, e.g.: [4, 2, 3]
        y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
            should be part of the return value of predict(x)
        s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
            should be part of the return value of predict(x)

        no return values
        '''

        for t in reversed(range(len(x))):
            ##########################
            # accumulate updates of W
            encoded_d = make_onehot(d[t], self.vocab_size)
            sig_out = (encoded_d - y[t])
            self.deltaW += np.outer(sig_out, s[t])

            # accumulate updates of V
            encoded_x = make_onehot(x[t], self.vocab_size)
            deri_net_in = np.multiply(s[t], np.subtract(np.ones(self.hidden_dims), s[t]))
            sig_in = np.multiply(np.matmul(np.transpose(self.W), sig_out), deri_net_in)
            self.deltaV += np.outer(sig_in, encoded_x)

            # accumulate updates of U
            self.deltaU += np.outer(sig_in, s[t - 1])

        ##########################

    def acc_deltas_np(self, x, d, y, s):
        '''
        accumulate updates for V, W, U
        standard back propagation

        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        for number prediction task, we do binary prediction, 0 or 1

        x	list of words, as indices, e.g.: [0, 4, 2]
        d	array with one element, as indices, e.g.: [0] or [1]
        y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
            should be part of the return value of predict(x)
        s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
            should be part of the return value of predict(x)

        no return values
        '''

        encoded_d = make_onehot(d[-1], self.vocab_size)

        # for t in reversed(range(len(x))):
        ##########################
        # accumulate updates of W
        sig_out = (encoded_d - y[len(x) - 1])
        self.deltaW += np.outer(sig_out, s[len(x) - 1])

        # accumulate updates of V
        encoded_x = make_onehot(x[len(x) - 1], self.vocab_size)
        deri_net_in = np.multiply(s[len(x) - 1], np.subtract(np.ones(self.hidden_dims), s[len(x) - 1]))
        sig_in = np.multiply(np.matmul(np.transpose(self.W), sig_out), deri_net_in)
        self.deltaV += np.outer(sig_in, encoded_x)

        # accumulate updates of U
        self.deltaU += np.outer(sig_in, s[len(x) - 1 - 1])


    ##########################


    def acc_deltas_bptt(self, x, d, y, s, steps):
        '''
        accumulate updates for V, W, U
        back propagation through time (BPTT)

        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time

        x		list of words, as indices, e.g.: [0, 4, 2]
        d		list of words, as indices, e.g.: [4, 2, 3]
        y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                should be part of the return value of predict(x)
        s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                should be part of the return value of predict(x)
        steps	number of time steps to go back in BPTT

        no return values
        '''

        for t in reversed(range(len(x))):
            #########################
            # accumulate updates of W
            encoded_d = make_onehot(d[t], self.vocab_size)
            sig_out = (encoded_d - y[t])
            self.deltaW += np.outer(sig_out, s[t])

            # accumulate updates of V and U
            encoded_x = make_onehot(x[t], self.vocab_size)
            deri_net_in = np.multiply(s[t], np.subtract(np.ones(self.hidden_dims), s[t]))
            sig_in = np.multiply(np.matmul(np.transpose(self.W), sig_out), deri_net_in)

            self.deltaV += np.outer(sig_in, encoded_x)
            self.deltaU += np.outer(sig_in, s[t - 1])

            for i in range(1, min(t, steps) + 1):
                encoded_x = make_onehot(x[t - i], self.vocab_size)
                deri_net_in = np.multiply(s[t - i], np.subtract(np.ones(self.hidden_dims), s[t - i]))

                sig_in = np.multiply(np.matmul(np.transpose(self.U), sig_in), deri_net_in)
                self.deltaV += np.outer(sig_in, encoded_x)
                self.deltaU += np.outer(sig_in, s[t - i - 1])
    #########################


    def acc_deltas_bptt_np(self, x, d, y, s, steps):
        '''
        accumulate updates for V, W, U
        back propagation through time (BPTT)

        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        for number prediction task, we do binary prediction, 0 or 1

        x	list of words, as indices, e.g.: [0, 4, 2]
        d	array with one element, as indices, e.g.: [0] or [1]
        y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                should be part of the return value of predict(x)
        s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                should be part of the return value of predict(x)
        steps	number of time steps to go back in BPTT

        no return values
        '''

    #########################
    # accumulate updates of W
        encoded_d = make_onehot(d[-1], self.vocab_size)
        sig_out = (encoded_d - y[len(x) - 1])
        self.deltaW += np.outer(sig_out, s[len(x) - 1])

        # accumulate updates of V and U
        encoded_x = make_onehot(x[len(x) - 1], self.vocab_size)
        deri_net_in = np.multiply(s[len(x) - 1], np.subtract(np.ones(self.hidden_dims), s[len(x) - 1]))
        sig_in = np.multiply(np.matmul(np.transpose(self.W), sig_out), deri_net_in)

        self.deltaV += np.outer(sig_in, encoded_x)
        self.deltaU += np.outer(sig_in, s[len(x) - 1 - 1])

        for i in range(1, min(len(x) - 1, steps) + 1):
            encoded_x = make_onehot(x[len(x) - 1 - i], self.vocab_size)
            deri_net_in = np.multiply(s[len(x) - 1 - i], np.subtract(np.ones(self.hidden_dims), s[len(x) - 1 - i]))

            sig_in = np.multiply(np.matmul(np.transpose(self.U), sig_in), deri_net_in)
            self.deltaV += np.outer(sig_in, encoded_x)
            self.deltaU += np.outer(sig_in, s[len(x)-1 - i - 1])


