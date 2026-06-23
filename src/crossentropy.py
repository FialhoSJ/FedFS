import numpy as np
import pymit as mit
from scipy import stats
from joblib import Parallel, delayed

import arrayoperations as ao


def get_features_ce(X, Y, init_probability=[0.5], classes=2,
                    selection_percentile=10, selection_probability=0.99,
                    max_sample_ce=100, max_steps_ce=10, alpha=1,
                    epsilon_tolerance=1e-5, current_step=1,
                     subsample_perc=-1, alpha_step=0, verbose=False,
                     early_stop=False, njobs=2, max_feat_per_sample_eval=50):

    N, num_features = X.shape

    if len(init_probability) == 1:
        prob_dist_features = np.ones(num_features) * init_probability[0]
    else:
        prob_dist_features = init_probability

    selected_features_samples = []
    old_prob = np.zeros_like(prob_dist_features)
    kstest_curr = 0
    kstest_prev = 0
    is_converged = False
    obj_fun = np.zeros((max_sample_ce, 2))

    for num_steps in np.arange(max_steps_ce):
        if verbose:
            print('Step: {}'.format(num_steps))
            print('Alpha: {}'.format(alpha))
            print('Num Feats: {}'.format(
                len(prob_dist_features[prob_dist_features > selection_probability])))
            print('Prob Feats: {}'.format(prob_dist_features))

        features_samples = (np.random.binomial(
            n=1, p=prob_dist_features,
            size=[max_sample_ce, num_features])).astype(int)

        if max_feat_per_sample_eval > 0:
            for i in range(features_samples.shape[0]):
                selected = np.where(features_samples[i] > 0)[0]
                if len(selected) > max_feat_per_sample_eval:
                    subsampled = np.random.choice(
                        selected, max_feat_per_sample_eval, replace=False)
                    features_samples[i] = 0
                    features_samples[i, subsampled] = 1

        if subsample_perc > 0:
            ssize = int(X.shape[0] * subsample_perc)
            data_idx = np.random.choice(X.shape[0], size=ssize, replace=True)
            XX = X[data_idx, :]
            YY = Y[data_idx]
        else:
            XX = X
            YY = Y

        if njobs > 1:
            results = Parallel(n_jobs=njobs, verbose=0)(
                delayed(parallel_obj_computation)(
                    i, XX[:, features_samples[i, :] > 0], YY, classes)
                for i in np.arange(features_samples.shape[0]))
            for res in results:
                if res is not None:
                    obj_fun[res[0], :] = res
        else:
            for i in np.arange(features_samples.shape[0]):
                fst = features_samples[i, :] > 0
                if np.any(fst):
                    outvec, maxVal = ao.merge_multiple_arrays(
                        XX[:, fst])
                    obj_fun[i, :] = [i, mit.H_cond(
                        YY, outvec, bins=[classes, maxVal])]

        np.sum(obj_fun[:, 0])
        obj_fun[:, 1] = np.nan_to_num(obj_fun[:, 1], nan=np.inf)

        perc = np.percentile(obj_fun[:, 1], selection_percentile)
        selected_features_samples = features_samples[
            obj_fun[obj_fun[:, 1] <= perc, 0].astype(int), :]

        np.copyto(old_prob, prob_dist_features)

        if selected_features_samples.shape[0] > 0:
            new_p = np.sum(selected_features_samples, axis=0) / \
                selected_features_samples.shape[0]
            if np.all(np.isreal(new_p)) and np.all(new_p >= 0) and np.all(new_p <= 1):
                prob_dist_features = (1 - alpha) * prob_dist_features + alpha * new_p
                np.clip(prob_dist_features, 1e-10, 1 - 1e-10, out=prob_dist_features)

        if early_stop:
            kstest_prev = kstest_curr
            kstest_curr = stats.ks_2samp(old_prob, prob_dist_features)[1]
            var = np.abs(kstest_curr - kstest_prev)
            if var <= epsilon_tolerance and kstest_curr > .995 and not is_converged:
                print("Convergence at {}!!!".format(num_steps))
                convergence_round = num_steps
                is_converged = True
                break

        alpha -= alpha_step

    selected_feats = np.where(prob_dist_features >= selection_probability)[0]

    return selected_feats, prob_dist_features, \
        np.sum(selected_features_samples, axis=0), \
        selected_features_samples.shape[0], num_steps, alpha


def parallel_obj_computation(i, X, Y, classes):
    if X.shape[1] > 0:
        outvec, maxVal = ao.merge_multiple_arrays(X)
        return [i, mit.H_cond(Y, outvec, bins=[classes, maxVal])]
    return None
