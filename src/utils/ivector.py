import numpy as np
import time
import pickle
import os

def compute_bw_stats(vX, GMM):
    nmix, ndim = GMM.means_.shape
    m = np.reshape(GMM.means_.T, (ndim * nmix, 1))
    idx_sv = np.reshape(np.matlib.repmat(range(0,nmix), ndim, 1).T, (ndim * nmix, 1))
    
    posterioriX = GMM.predict_proba(vX) + 1e-18*np.random.rand(vX.shape[0],nmix)
    N = posterioriX.sum(0)
    F = np.matmul(vX.T,posterioriX)
    
    F = np.reshape(F.T, (ndim * nmix, 1))
    F = F - np.multiply(N[idx_sv],m) # centered first order stats

    return N, F

def extract_ivector(statVector, ubmData, T_matrix):
    nmix, ndim  = ubmData.means_.shape
    S = np.reshape(ubmData.covariances_.T, (ndim * nmix, 1))
    idx_sv = np.reshape(np.matlib.repmat(range(0,nmix), ndim, 1).T, (ndim * nmix, 1))


    tv_dim = T_matrix.shape[0]
    I = np.eye(tv_dim)
    T_invS =  np.divide(T_matrix, np.matlib.repmat(S.T,tv_dim,1))
    N = statVector[:nmix]
    F = statVector[nmix:]

    L = I +  np.matmul(np.multiply(T_invS, N[idx_sv].T), T_matrix.T)
    B = np.matmul( T_invS,F)
    x = np.matmul(np.linalg.pinv(L),B)

    return x

def train_tv_space(data, UBM, tv_dim, nIter, filesave):
    randInit = False
    epsMIN = 1e-3
    epsMAX = 1e4
    epsChange = 1e-5
    notChangeIter = 0
    maxNotChange = 10
    nmix, ndim = UBM.means_.shape
    S = np.reshape(UBM.covariances_, (ndim * nmix, 1));
    iniIter = 0
    if (os.path.exists(filesave)):
        with open(filesave, 'rb') as f:
            T = pickle.load(f)
            iniIter = pickle.load(f)
            iterMeanAnt = pickle.load(f)
            iterStdAnt = pickle.load(f)
            notChangeIter = pickle.load(f)
    else:
        print('\n\nRandomly initializing T matrix ...\n\n');
        randInit = True
        iterMeanAnt = 1e3
        iterStdAnt = 1e3
        # suggested in JFA cookbook
        T = np.random.rand(tv_dim, ndim * nmix) * S.sum(0) * 0.001;
    
    N, F = load_data(data, ndim, nmix)
        
    print('Re-estimating the total subspace with {:} factors ...\n'.format(tv_dim));
    for iter in range(iniIter,nIter):
        print('EM iter#: {:} \t'.format(iter))
        tim = time.time()
        LU, RU = expectation_tv(T, N, F, S, tv_dim, nmix, ndim);
        Tnew = maximization_tv(LU, RU, ndim, nmix);
        iterMean = np.absolute(Tnew - T).mean()
        iterStd = np.absolute(Tnew - T).std()
        tim = (time.time() - tim);
        print('[elaps = {:.2f} s, mean: {:3.2e}, std: {:3.2e}, notChange {:}]\n'.format(tim, iterMean, iterStd,notChangeIter))
        T = Tnew
        
        if (((not randInit) and (iterMean < epsMIN) and (iterStd < epsMAX)) or (notChangeIter >= 10)):
            print('Estimantion convergence')
            with open(filesave, 'wb') as f:
                pickle.dump(T,f)
                pickle.dump(iter,f)
                pickle.dump(iterMeanAnt,f)
                pickle.dump(iterStdAnt,f)
                pickle.dump(notChangeIter,f)
            break
        if ( (np.absolute(iterMean - iterMeanAnt) < epsChange) and (np.absolute(iterStd - iterStdAnt) < epsChange) ):
            notChangeIter += 1
        else:
            notChangeIter = 0
        if ((not randInit) and (iterStd > epsMAX)):
            randInit = True
            T = np.random.rand(tv_dim, ndim * nmix) * S.sum(0) * 0.001;
            iterMeanAnt = 1e3
            iterStdAnt = 1e3
            notChangeIter = 0
        if (randInit):
            randInit = False
            
        iterMeanAnt = iterMean
        iterStdAnt = iterStd
        with open(filesave, 'wb') as f:
            pickle.dump(T,f)
            pickle.dump(iter,f)
            pickle.dump(iterMeanAnt,f)
            pickle.dump(iterStdAnt,f)
            pickle.dump(notChangeIter,f)
        
    return T

def load_data(datalist, ndim, nmix):
    nfiles = datalist.shape[0]
    N = np.zeros((nfiles, nmix))
    F = np.zeros((nfiles, ndim * nmix))
    for file in range(0,nfiles):
        N[file, :] = datalist[file,:nmix]
        F[file, :] = datalist[file,nmix:]
    
    return N, F

def expectation_tv(T, N, F, S, tv_dim, nmix, ndim):
# compute the posterior means and covariance matrices of the factors 
# or latent variables
    idx_sv = np.reshape(np.matlib.repmat(range(0,nmix), ndim, 1).T, (ndim * nmix, 1))
    nfiles = F.shape[0];

    LU = np.zeros((nmix, tv_dim, tv_dim))
    RU = np.zeros((tv_dim, nmix * ndim))
    I = np.eye(tv_dim)
    T_invS =  np.divide(T, np.matlib.repmat(S.T,tv_dim,1))

    parts = 250; # modify this based on your resources
    nbatch = int(np.floor( nfiles/parts + 0.999999 ))
    for batch in range(0,nbatch):
        start = 0 + batch * parts
        fin = min((batch +1) * parts, nfiles)
        len = fin - start;
        index = range(start,fin)
        N1 = N[index, :]
        F1 = F[index, :]
        Ex = np.zeros((tv_dim, len))
        Exx = np.zeros((tv_dim, tv_dim, len))
        for ix in range(0,len):
            L = I +  np.matmul(np.multiply(T_invS, N1[ix, idx_sv].T), T.T)
            Cxx = np.linalg.pinv(L) # this is the posterior covariance Cov(x,x)
            B = np.matmul( T_invS,F1[ix, :])
            B.shape = (tv_dim,1)
            ExV = np.matmul(Cxx ,B) # this is the posterior mean E[x]
            ExV.shape = (tv_dim,)
            Ex[:, ix] = ExV
            ExV.shape = (tv_dim,1)
            Exx[:, :, ix] = Cxx + np.matmul(ExV,ExV.T)
            
        RU = RU + np.matmul(Ex,F1)
        for mix in range(0,nmix):
            tmp = np.multiply(Exx, N1[:, mix])
            LU[mix,:,:] = LU[mix,:,:] + tmp.sum(2);
    return LU, RU

def maximization_tv(LU, RU, ndim, nmix):
# ML re-estimation of the total subspace matrix or the factor loading
# matrix
    for mix in range(0,nmix):
        idx = range( mix* ndim, (mix+1)* ndim);
        RU[:, idx] = np.linalg.solve(LU[mix,:,:],RU[:, idx])

    return RU