import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.misc import comb
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
# from joblib import Parallel, delayed
# from dask import delayed
# from dask.distributed import Client, LocalCluster

# set up global variables to be shared across cores (read-only)
# DESIGN = None
# RESPONSE = None
# PRIORS = None
# REGULATORS = None
# PRIORS = None
# PRIOR_WEIGHT = None

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
                if alpha <= p[j]*lamS:
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

    def get_response_for_targets(self, response, targets):
        responses = {}
        for target in targets:
            response_one_target = {}
            for k in range(len(response)):
                if target in response[k]:
                    response_one_target[k] = response[k][target].values.reshape(-1, 1)
            responses[target] = response_one_target
        return(responses)


    def run(self, design, response, targets, regulators, priors=None, prior_weight=1, cluster_id=None):
        '''

        '''
        # split response matrix...
        responses = self.get_response_for_targets(response, targets)
        # create a function call that takes in a particular target
        regression_call = lambda target: run_regression_EBIC(design, responses[target], target, regulators, priors, prior_weight)
        # serial_result = [regression_call(target) for target in targets]

        del response # free memory

        if cluster_id is None:
            results = [regression_call(target) for target in targets]

        else:
            import os
            from ipyparallel import Client

            client = Client(cluster_id=cluster_id)
            lview = client.load_balanced_view()
            lview.map(os.chdir, [os.getcwd()]*len(client.ids))

            # initiate list of results for each gene
            results = []
            # split targets in chunks of maximum length of a hundred
            # this significantly saves up memory...
            # in case of very limited memory, one can change this to an even lower number
            targets_chunks = [targets[idx:idx + 100] for idx in xrange(0, len(targets), 100)]

            # parallelize across processes in ipcluster
            for targets_chunk in targets_chunks:
                # map_async from balanced_view obj will schedule regression_call for each target to a different process
                partial_results = lview.map_async(regression_call, targets_chunk)
                # tell us how long each 100 genes take to run
                partial_results.wait_interactive()
                # actually gather results
                partial_results = partial_results.get()
                # save results for this chunk in results
                results.append(partial_results)
                # clear up caches from previous tasks from both the hub and the local client
                client.purge_everything()
            # close connection to socket? re-open in next bootstrap anyway
            client.close()
            # join partial_results lists into one list of results
            results = [result for partial_results in results for result in partial_results]


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


def format_prior(priors, gene, TFs, tasks, prior_weight):
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


def run_regression_EBIC(design, response, target, regulators, priors, prior_weight):
    '''

    '''
    # remove self regulation
    tfs = [tf for tf in regulators if tf != target]

    tasks = response.keys()

    X = [design[k][tfs] for k in tasks]
    Y = response.values()
    prior = format_prior(priors, target, tfs, tasks, float(prior_weight))

    del design, response, priors

    if len(tasks) > 1:

        n_tasks = len(X)
        n_preds = X[0].shape[1]
        n_samples = [X[k].shape[0] for k in range(n_tasks)]

        ###### EBIC ######
        Cs = np.logspace(np.log10(0.01), np.log10(10), 20)[::-1]
        Ss = np.linspace(0.51, 0.99, 10)[::-1]
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

        for k in tasks:
            nonzero = outW[:,k] != 0
            if nonzero.sum() > 0:
                chosen_regulators = np.asarray(tfs)[nonzero != 0]
                output[k] = final_weights(X[k][:, nonzero], Y[k], chosen_regulators, target)

        return(output)


        ######################################################
        ######################## DASK ########################
        ######################################################
        # cluster = LocalCluster(n_workers = 3)
        # client = Client(cluster)
        #
        # # "scatter" large datasets across cores at once, and pass pointer to function!
        # [design_future] = client.scatter([design])
        # [response_future] = client.scatter([response])
        # [priors_future] = client.scatter([priors])
        # # run regression in parallel
        # jobs = [client.submit(run_regression_EBIC, design_future, response_future, regulators, target, priors_future, prior_weight) for target in targets]
        # results = client.gather(jobs, asynchronous=True)
        ######################################################
        ######################## DASK ########################
        ######################################################


        ############################################
        ################ joblib ####################
        ############################################
        # jobs = (delayed(run_regression_EBIC_ipp)(target) for target in targets)
        # results = Parallel(cluster_id=cluster_id, verbose=5, backend='threading')(jobs)
        ############################################
        ################ joblib ####################
        ############################################


# def run_regression_EBIC_ipp(target):
#     '''
#
#     '''
#
#     X = []; Y = [];
#     tasks = []; prior = []
#     # remove self regulation
#     TFs = [tf for tf in REGULATORS if tf != target]
#
#     for k in range(len(DESIGN)):
#         if target in RESPONSE[k]:
#             X.append(DESIGN[k][TFs])
#             Y.append(RESPONSE[k][target].values.reshape(-1, 1))
#             tasks.append(k)
#
#     prior = format_prior(PRIORS, target, TFs, tasks, float(PRIOR_WEIGHT))
#
#     if len(tasks) > 1:
#
#         n_tasks = len(X)
#         n_preds = X[0].shape[1]
#         n_samples = [X[k].shape[0] for k in range(n_tasks)]
#
#         ###### EBIC ######
#         Cs = np.logspace(np.log10(0.01), np.log10(10), 20)[::-1]
#         Ss = np.linspace(0.51, 0.99, 10)[::-1]
#         lamBparam = np.sqrt((n_tasks * np.log(n_preds))/np.mean(n_samples))
#
#         model = AMuSR_OneGene(n_tasks, n_preds)
#         X, Y = model.preprocess_data(X, Y)
#         C, D = model.covariance_update_terms(X, Y)
#         S = np.zeros((n_preds, n_tasks))
#         B = np.zeros((n_preds, n_tasks))
#
#         min_ebic = float('Inf')
#
#         for c in Cs:
#             tmp_lamB = c * lamBparam
#             for s in Ss:
#                 tmp_lamS = s * tmp_lamB
#                 W, S, B = model.fit(X, Y, tmp_lamB, tmp_lamS, C, D, S, B, prior)
#                 ebic_score = ebic(X, Y, W, n_tasks, n_samples, n_preds)
#                 if ebic_score < min_ebic:
#                     min_ebic = ebic_score
#                     lamB = tmp_lamB
#                     lamS = tmp_lamS
#                     outW = W
#
#         ###### RESCALE WEIGHTS ######
#         output = {}
#
#         for k in tasks:
#             nonzero = outW[:,k] != 0
#             if nonzero.sum() > 0:
#                 chosen_regulators = np.asarray(TFs)[nonzero != 0]
#                 output[k] = final_weights(X[k][:, nonzero], Y[k], chosen_regulators, target)
#
#         return(output)


# def run_regression_multiple_genes(design, response, regulators, targets, priors, prior_weight):
#
#     targets = ['BSU02100', 'BSU05340', 'BSU24010', 'BSU24040'] # test
#     import os
#
#     y = 3
#     client = Client() # processes=False
#     #client.map(square, range(10))
#     # y = [client.submit(test_model, x) for x in np.linspace(0, 1)]
#     # y = client.gather(y)  # collect the results
#     # print(y)
#
#     jobs = (delayed(run_regression_EBIC)(design, response,
#                  regulators, target, priors, prior_weight) for target in targets)
#
#     # lazy_values = [delayed(square)(future, x) for x in [1,2,3]]
#     futures = client.compute(jobs)
#     # futures = c.compute(lazy_values)
#     # regulators = client.scatter(y, broadcast=True)
#     print(futures)
#     # results = client.map(square, range(10), regulators)
#     # results = client.gather(results)
#     print(futures.result())
#
#     # from ipyparallel import Client
#     # client = Client()
#     # lview = client[:]
#     # lview.map(os.chdir, [os.getcwd()]*len(client.ids))
#     # lview.push(dict(design = design, response = response,
#     #     regulators = regulators, priors = priors, prior_weight = prior_weight))
#     # print(client[0]['regulators'])
#     # results = client[:].apply_sync(run_regression_EBIC_ipp, targets)
#     # return results
#
# def square(y, x):
#
#     return (x ** 2, y ** 2)
