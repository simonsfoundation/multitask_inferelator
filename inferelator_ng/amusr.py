import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.misc import comb
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import utils
import os


class AMuSR_OneGene:

    max_iter = 1000
    tolerance = 1e-2

    def __init__(self, n_tasks, n_features):

        self.n_tasks = n_tasks
        self.n_features = n_features


    def preprocess_data(self, X, Y):
        """
        center and standardize input X and Y data (z-score)
        """

        for k in range(self.n_tasks):
            X[k] = StandardScaler().fit_transform(X[k])
            Y[k] = StandardScaler().fit_transform(Y[k])

        return((X, Y))

    def covariance_update_terms(self, X, Y):
        """
        returns C and D, containing terms for covariance update for OLS fit
        C: transpose(X_j)*Y for each feature j
        D: transpose(X_j)*X_l for each feature j for each feature l
        reference: Friedman, Hastie, Tibshirani, 2010 in Journal of Statistical Software
        Regularization Paths for Generalized Linear Models via Coordinate Descent.
        """

        C = np.zeros((self.n_tasks, self.n_features))
        D = np.zeros((self.n_tasks, self.n_features, self.n_features))

        for k in range(self.n_tasks):
            C[k] = np.dot(Y[k].transpose(), X[k])
            D[k] = np.dot(X[k].transpose(), X[k])

        return((C, D))


    def updateS(self, C, D, B, S, lamS, prior):
        """
        returns updated coefficients for S (predictors x tasks)
        lasso regularized -- using cyclical coordinate descent and
        soft-thresholding
        """
        # update each task independently (shared penalty only)
        for k in range(self.n_tasks):
            c = C[k]; d = D[k]
            b = B[:,k]; s = S[:,k]
            p = prior[:,k]
            # cycle through predictors
            for j in range(self.n_features):
                # set sparse coefficient for predictor j to zero
                s_tmp = deepcopy(s)
                s_tmp[j] = 0.
                # calculate next coefficient based on fit only
                if d[j,j] == 0:
                    alpha = 0
                else:
                    alpha = (c[j]-np.sum((b+s_tmp)*d[j]))/d[j,j]
                # lasso regularization
                if abs(alpha) <= p[j]*lamS:
                    s[j] = 0.
                else:
                    s[j] = alpha-(np.sign(alpha)*p[j]*lamS)
            # update current task
            S[:,k] = s

        return(S)

    def updateB(self, C, D, B, S, lamB, prior):
        """
        returns updated coefficients for B (predictors x tasks)
        block regularized (l_1/l_inf) -- using cyclical coordinate descent and
        soft-thresholding on the l_1 norm across tasks
        reference: Liu et al, ICML 2009. Blockwise coordinate descent procedures
        for the multi-task lasso, with applications to neural semantic basis discovery.
        """
        #p = prior.min(axis=1)
        # cycles through predictors
        for j in range(self.n_features):
            # initialize next coefficients
            alphas = np.zeros(self.n_tasks)
            # update tasks for each predictor together
            for k in range(self.n_tasks):
                # get task covariance update terms
                c = C[k]; d = D[k]
                # get previous block-sparse and sparse coefficients
                b = B[:,k]; s = S[:,k]
                # set block-sparse coefficient for feature j to zero
                b_tmp = deepcopy(b)
                b_tmp[j] = 0.
                # calculate next coefficient based on fit only
                if d[j,j] == 0:
                    alphas[k] = 0
                else:
                    alphas[k] = (c[j]-np.sum((b_tmp+s)*d[:,j]))/d[j,j]
            # set all tasks to zero if l1-norm less than lamB
            if np.linalg.norm(alphas, 1) <= lamB:
                B[j,:] = np.zeros(self.n_tasks)
            # regularized update for predictors with larger l1-norm
            else:
                # find number of coefficients that would make l1-norm greater than penalty
                indices = np.abs(alphas).argsort()[::-1]
                sorted_alphas = alphas[indices]
                m_star = np.argmax((np.abs(sorted_alphas).cumsum()-lamB)/(np.arange(self.n_tasks)+1))
                # initialize new weights
                new_weights = np.zeros(self.n_tasks)
                # keep small coefficients and regularize large ones (in above group)
                for k in range(self.n_tasks):
                    idx = indices[k]
                    if k > m_star:
                        new_weights[idx] = sorted_alphas[k]
                    else:
                        sign = np.sign(sorted_alphas[k])
                        update_term = np.sum(np.abs(sorted_alphas)[:m_star+1])-lamB
                        new_weights[idx] = (sign/(m_star+1))*update_term
                # update current predictor
                B[j,:] = new_weights

        return(B)

    def fit(self, X, Y, lamB=0., lamS=0., C=None, D=None, S=None, B=None, prior=None):
        """
        Fits regression model in which the weights matrix W (predictors x tasks)
        is decomposed in two components: B that captures block structure across tasks
        and S that allows for the differences.
        reference: Jalali et al., NIPS 2010. A Dirty Model for Multi-task Learning.
        """
        # calculate covariance update terms if not provided
        if C is None or D is None:
            C, D = self.covariance_update_terms(X, Y)
        # if S and B are provided -- warm starts -- will run faster
        if S is None or B is None:
            S = np.zeros((self.n_features, self.n_tasks))
            B = np.zeros((self.n_features, self.n_tasks))
        if prior is None:
            prior = np.ones((self.n_features, self.n_tasks))
        # initialize W
        W = S + B
        for n_iter in range(self.max_iter):
            # save old values of W (to check convergence)
            W_old = deepcopy(W)
            # update S and B coefficients
            S = self.updateS(C, D, B, S, lamS, prior)
            B = self.updateB(C, D, B, S, lamB, prior)
            W = S + B
            # update convergence criteria
            update = np.max(np.abs(W-W_old))
            if update < self.tolerance:
                break
        # weights matrix (W) is the sum of a sparse (S) and a block-sparse (B) matrix
        W = S + B
        # set small values of W to zero
        # since we don't run the algorithm until update equals zero
        W[np.abs(W) < 0.1] = 0

        return(W, S, B)



class AMuSR_regression:

    '''

    '''

    def __init__(self):
        pass

    def format_weights(self, df, col, targets, regs):

        df[col] = pd.to_numeric(df[col])

        out = pd.pivot_table(df, index = 'target',
                                 columns = 'regulator',
                                 values = col,
                                 fill_value = 0.)
        del out.columns.name
        del out.index.name

        out = pd.concat([out,
                    pd.DataFrame(0., index = out.index,
                            columns = np.setdiff1d(regs, out.columns))], axis = 1)
        out = pd.concat([out,
                    pd.DataFrame(0., index = np.setdiff1d(targets, out.index),
                            columns = out.columns)])
        out = out.loc[targets, regs]

        return(out)

    def format_prior(self, priors, gene, TFs, tasks, prior_weight):
        '''
        Returns priors for one gene (numpy matrix TFs by tasks)
        '''
        if priors is None:
            priors_out = None
        else:
            priors_out = []
            for k in tasks:
                prior = priors[k]
                prior = prior.loc[gene, TFs].replace(np.nan, 0)
                prior = (prior != 0).astype(int)
                prior /= prior_weight
                prior[prior == 0] = 1.
                prior = prior/prior.sum()*len(prior)
                priors_out.append(prior)
            priors_out = np.transpose(np.asarray(priors_out))
        return(priors_out)


    def run(self, design, response, targets, regulators,
                    kvs, rank, ownCheck,
                    cluster_id=None, priors=None, prior_weight=1):
        '''

        '''
        prior_weight = float(prior_weight)
        results = []

        for gene in targets:
            if ownCheck.next():
                X = []; Y = []; tasks = []; prior = []
                TFs = [tf for tf in regulators if tf != gene] # remove self regulation

                for k in range(len(design)):
                    if gene in response[k]:
                        X.append(design[k][TFs])
                        Y.append(response[k][gene].values.reshape(-1, 1))
                        tasks.append(k)
                if len(X) > 1:
                    prior = self.format_prior(priors, gene, TFs, tasks, prior_weight)
                    results.append(run_regression_EBIC(X, Y, TFs, tasks, gene, prior))

        kvs.put('plist %d'%rank, (rank, results))

        if rank == 0:
            results=[]
            workers=int(os.environ['SLURM_NTASKS'])
            for p in range(workers):
                wrank, ps = kvs.get('plist %d'%p)
                results.extend(ps)
            print ('rank 0 worker got final results of length: ', len(results))
            utils.kvsTearDown(kvs, rank)

            weights = []
            rescaled_weights = []

            for k in range(len(design)):
                results_k = []
                for res in results:
                    try:
                        results_k.append(res[k])
                    except:
                        pass

                results_k = pd.concat(results_k)
                weights_k = self.format_weights(results_k, 'weights', targets, regulators)
                rescaled_weights_k = self.format_weights(results_k, 'resc_weights', targets, regulators)
                rescaled_weights_k[rescaled_weights_k < 0.] = 0

                weights.append(weights_k)
                rescaled_weights.append(rescaled_weights_k)

            return((weights, rescaled_weights))
        else:
            return(None, None)


def sum_squared_errors(X, Y, W, k):
    '''
    Get RSS for a particular task 'k'
    '''
    return(np.sum((Y[k].T-np.dot(X[k], W[:,k]))**2))


def ebic(X, Y, W, n_tasks, n_samples, n_preds, gamma=1):
    '''
    Calculate EBIC for each task, and take the mean
    '''
    EBIC = []

    for k in range(n_tasks):

        n = n_samples[k]
        nonzero_pred = (W[:,k] != 0).sum()

        RSS = sum_squared_errors(X, Y, W, k)
        BIC_penalty = nonzero_pred * np.log(n)
        BIC_extension = 2 * gamma * np.log(comb(n_preds, nonzero_pred))
        EBIC.append((n * np.log(RSS/n)) + BIC_penalty + BIC_extension)

    EBIC = np.mean(EBIC)

    return(EBIC)



def final_weights(X, y, TFs, gene):
    """
    returns reduction on variance explained for each predictor
    (model without each predictor compared to full model)
    see: Greenfield et al., 2013. Robust data-driven incorporation of prior
    knowledge into the inference of dynamic regulatory networks.
    """
    n_preds = len(TFs)
    # linear fit using sklearn
    ols = LinearRegression()
    ols.fit(X, y)
    # save weights and initialize rescaled weights vector
    weights = ols.coef_[0]
    resc_weights = np.zeros(n_preds)
    # variance of residuals (full model)
    var_full = np.var((y - ols.predict(X))**2)
    # when there is only one predictor
    if n_preds == 1:
        resc_weights[0] = 1 - (var_full/np.var(y))
    # remove each at a time and calculate variance explained
    else:
        for j in range(len(TFs)):
            X_noj = X[:, np.setdiff1d(range(n_preds), j)]
            ols = LinearRegression()
            ols.fit(X_noj, y)
            var_noj = np.var((y - ols.predict(X_noj))**2)
            resc_weights[j] = 1 - (var_full/var_noj)
    # format output
    out_weights = pd.DataFrame([TFs, [gene]*len(TFs), weights, resc_weights]).transpose()
    out_weights.columns = ['regulator', 'target', 'weights', 'resc_weights']
    return(out_weights)


def run_regression_EBIC(X, Y, TFs, tasks, gene, prior):
    '''

    '''
    n_tasks = len(X)
    n_preds = X[0].shape[1]
    n_samples = [X[k].shape[0] for k in range(n_tasks)]

    ###### EBIC ######
    Cs = np.logspace(np.log10(0.01), np.log10(10), 20)[::-1]
    Ss = np.linspace((1.0/n_tasks)+0.01, 0.99, 10)[::-1] # in paper I used 0.51 as minimum for all networks
    lamBparam = np.sqrt((n_tasks * np.log(n_preds))/np.mean(n_samples))

    model = AMuSR_OneGene(n_tasks, n_preds)
    X, Y = model.preprocess_data(X, Y)
    C, D = model.covariance_update_terms(X, Y)
    S = np.zeros((n_preds, n_tasks))
    B = np.zeros((n_preds, n_tasks))

    min_ebic = float('Inf')

    for c in Cs:
        tmp_lamB = c * lamBparam
        for s in Ss:
            tmp_lamS = s * tmp_lamB
            W, S, B = model.fit(X, Y, tmp_lamB, tmp_lamS, C, D, S, B, prior)
            ebic_score = ebic(X, Y, W, n_tasks, n_samples, n_preds)
            if ebic_score < min_ebic:
                min_ebic = ebic_score
                lamB = tmp_lamB
                lamS = tmp_lamS
                outW = W

    ###### RESCALE WEIGHTS ######
    output = {}

    for kx, k in enumerate(tasks):
        nonzero = outW[:,kx] != 0
        if nonzero.sum() > 0:
            cTFs = np.asarray(TFs)[outW[:,kx] != 0]
            output[k] = final_weights(X[kx][:, nonzero], Y[kx], cTFs, gene)
    return(output)
