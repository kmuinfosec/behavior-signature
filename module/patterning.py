import numpy as np

from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm


class GMM_Pattering:
    def __init__(self, ignore_idx=[], random_seed=43, covariance_type='full', max_iter=2000, n_components=10,
                 reg_covar=1e-6, tol=1e-3):
        self.reg_covar = reg_covar
        self.tol = tol
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.ignore_idx = ignore_idx
        self.models = []
        self.confidence = 0
        self.random_seed = random_seed
        self.table = {}

    def fit(self, data):
        np_data = np.array(data)
        print("Model Fitting...")
        for idx in tqdm(range(len(data[0]))):
            if idx in self.ignore_idx:
                tmp_vgm = GaussianMixture(n_components=self.n_components, max_iter=self.max_iter,
                                          random_state=self.random_seed,
                                          covariance_type=self.covariance_type, reg_covar=self.reg_covar, tol=self.tol)
                self.models.append(tmp_vgm)
                continue
            tmp_data = np_data[:, idx].reshape(-1, 1)
            tmp_vgm = GaussianMixture(n_components=self.n_components, max_iter=self.max_iter,
                                      random_state=self.random_seed,
                                      covariance_type=self.covariance_type, reg_covar=self.reg_covar, tol=self.tol)
            tmp_vgm.fit(tmp_data)
            self.models.append(tmp_vgm)
            if self.covariance_type == 'full':
                self.table[idx] = {i: {"mean": tmp_vgm.means_[i][0], "std": tmp_vgm.covariances_[i][0][0] ** 0.5} for i
                                   in range(self.n_components)}
            elif self.covariance_type == 'spherical':
                self.table[idx] = {i: {"mean": tmp_vgm.means_[i][0], "std": tmp_vgm.covariances_[i] ** 0.5} for i in
                                   range(self.n_components)}

    def transform(self, data, confidence=2.58):
        self.confidence = confidence
        np_data = np.array(data)
        ret_data = np.empty_like(np_data, dtype='<U12')
        #         print("Data Transforming...")
        for idx in range(len(data[0])):
            if idx in self.ignore_idx:
                ret_data[:, idx] = np.array(
                    list(map(lambda x: chr(int(x) + 65).zfill(2), np_data[:, idx].astype('<U12'))))
                continue
            tmp_data = np_data[:, idx].reshape(-1, 1)
            pred = self.models[idx].predict(tmp_data).astype('<U12')
            for p_idx in range(len(pred)):
                tmp_pred = int(pred[p_idx])
                tmp_mean = self.table[idx][tmp_pred]['mean']
                tmp_std = self.table[idx][tmp_pred]['std']
                tmp_pred = chr(tmp_pred + 65)
                if tmp_mean - (self.confidence * tmp_std) > tmp_data[p_idx]:
                    pred[p_idx] = f'-{tmp_pred}'
                elif tmp_data[p_idx] > tmp_mean + (self.confidence * tmp_std):
                    pred[p_idx] = f'+{tmp_pred}'
                else:
                    pred[p_idx] = tmp_pred.zfill(2)
            ret_data[:, idx] = pred
        return ret_data

    def tokenize(self, data):
        token = list(map(lambda x: ''.join(x), data))
        return token

    def fit_transform(self, data, confidence=2.58):
        self.fit(data)
        return self.transform(data, confidence)

    def fit_transform_tokenize(self, data, confidence=2.58):
        tmp_data = self.fit_transform(data, confidence)
        return self.tokenize(tmp_data)

    def transform_tokenize(self, data, confidence=2.58):
        tmp_data = self.transform(data, confidence)
        return self.tokenize(tmp_data)