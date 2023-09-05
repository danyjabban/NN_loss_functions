import argparse
import numpy as np
import nn_utils as utils


def get_mnist_data():
    loaded = np.load("mnist_dat.npz")
    X, y = loaded["X"], loaded["y"]
    
    # make one-hot
    Y = np.zeros((X.shape[0], 10))
    Y[np.arange(Y.shape[0]), y] = 1

    valperc = 0.1
    nval = int(Y.shape[0] * valperc)

    val_X, val_Y = X[-nval:], Y[-nval:]
    X, Y = X[:-nval], Y[:-nval]
    return X, Y, val_X, val_Y


def eval_perf(theta, X, Y):
    Xs, Ys = batchify(X, Y, 10) # bsz=10 so we don't truncate on val
    total_loss, ncorrect, ntotal = 0, 0, 0
    for i, X_i in enumerate(Xs):
        Y_i = Ys[i]
        scores = utils.score(X_i, theta)
        preds = scores.argmax(1)
        golds = Y_i.argmax(1)
        ncorrect += (preds == golds).sum()
        losses = loss_fn(X_i, Y_i, theta)
        total_loss += losses.sum()        
        ntotal += Y_i.shape[0]
    return total_loss/ntotal, ncorrect/ntotal


def batchify(X, Y, bsz):
    ntrunc = X.shape[0] // bsz * bsz
    nbatch = ntrunc // bsz if bsz > 0 else 1
    Xs, Ys = np.split(X[:ntrunc], nbatch), np.split(Y[:ntrunc], nbatch)
    return Xs, Ys


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nepochs", type=int, default=20)
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="xent", choices=["xent", "mse", "softmse", "myloss"])
    args = parser.parse_args()

    np.random.seed(59003)
    X, Y, val_X, val_Y = get_mnist_data()
    # add bias terms
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    val_X = np.hstack((val_X, np.ones((val_X.shape[0], 1))))

    if args.loss == "xent":
        loss_fn, grad_theta_fn = utils.xent, utils.grad_theta_xent
    elif args.loss == "mse":
        loss_fn, grad_theta_fn = utils.mse, utils.grad_theta_mse
    elif args.loss == "softmse":
        loss_fn, grad_theta_fn = utils.softmse, utils.grad_theta_softmse
    else:
        loss_fn, grad_theta_fn = utils.myloss, utils.grad_theta_myloss

    # initialize theta
    theta = np.zeros((Y.shape[1], X.shape[1])) # K x D_1

    eta, bsz = args.eta, args.bsz
    for epoch in range(1, args.nepochs+1):
        randperm = np.random.permutation(X.shape[0])
        X, Y = X[randperm], Y[randperm]
        Xs, Ys = batchify(X, Y, bsz)

        total_loss, nexamples = 0, 0
        for i, X_i in enumerate(Xs):
            Y_i = Ys[i]
            nex = Y_i.shape[0]
            losses = loss_fn(X_i, Y_i, theta)
            total_loss += losses.sum()
            grad = grad_theta_fn(X_i, Y_i, theta)
            grad /= bsz # average over all the gradients in the batch
            theta = theta - eta * grad/nex
            nexamples += nex

        print("epoch {} | train_loss {:.4f} | eta {:.5f}".format(
            epoch, total_loss/nexamples, eta))

        val_loss, val_acc = eval_perf(theta, val_X, val_Y)
        print("epoch {} | val_loss {:.4f} | val_acc {:.3f}".format(
            epoch, val_loss, val_acc))
