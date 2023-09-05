import numpy as np

def logsumexp(Z, axis=1):
    """
    Z - an ndarray
    axis - the dimension over which to logsumexp
    returns: 
        logsumexp over the axis'th dimension; returned tensor has same ndim as Z
    """
    maxes = np.max(Z, axis=axis, keepdims=True)
    return maxes + np.log(np.exp(Z - maxes).sum(axis, keepdims=True))


def score(X, theta):
    """
    X - bsz x D_1
    theta - K x D_1
    returns: bsz x K
    """
    return np.matmul(X, theta.transpose())



def xent(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """
    s = score(X, theta)
    lgsum = logsumexp(s)
    a = (np.multiply(Y, s).sum(axis=1)).reshape(-1,1)
    loss = lgsum - a
    return loss



def grad_theta_xent(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of xent(X, Y, theta).sum() wrt theta
    """
    s = score(X, theta)
    grad = np.matmul((np.exp(s)/np.exp(logsumexp(s)) - Y).T, X) 
    return grad


def mse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """   
    
    loss = (((1/Y.shape[1])*(Y - score(X,theta))**2).sum(axis=1)).reshape(-1,1)
    return loss



def grad_theta_mse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of mse(X, Y, theta).sum() wrt theta
    """   
    grad = (-2/Y.shape[1])*np.matmul((Y-score(X,theta)).T, X)
    return grad


def softmax(s):
    sum = logsumexp(s)
    return np.exp(s)/(np.exp(sum))


def softmse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """
    s = score(X, theta)
    sm = softmax(s)
    loss = ((1/Y.shape[1])*((Y-sm)**2).sum(axis=1)).reshape(-1,1)
    return loss


def grad_theta_softmse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of softmse(X, Y, theta).sum() wrt theta
    """
    s = score(X, theta)
    sm = softmax(s)
    # b x k x 1 
    a = ((-2/Y.shape[1])*(Y-sm)*sm).reshape(Y.shape[0], Y.shape[1], 1)

    # 1 x k x k identity matrix
    I = (np.identity(Y.shape[1])).reshape(1,Y.shape[1], Y.shape[1])

    # b x k x 1 softmax matrix with additional dimension 
    sm_reshape = sm.reshape(sm.shape[0], sm.shape[1], 1)

    # b x k x k matrix 
    sm_i = I - sm_reshape

    # b x k x k matrix 
    c = a * sm_i

    # once we sum on on axis=1 we get the b x k matrix of derivatives of snk
    s_der = c.sum(axis=1)
    # k x D_1 matrix of derivatives of theta
    grad = np.matmul(s_der.T,X)
    return grad



def myloss(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """   
    loss = (((1/Y.shape[1])*(Y - np.exp(score(X,theta)))**2).sum(axis=1)).reshape(-1,1)
    return loss  


def grad_theta_myloss(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of myloss(X, Y, theta).sum() wrt theta
    """   
    grad = (-2/Y.shape[1])*np.matmul((Y*np.exp(score(X,theta))-np.exp(2*score(X,theta))).T, X)
    return grad


