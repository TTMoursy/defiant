import jax
jax.config.update("jax_enable_x64", True) # switch to 64-bit precision
import jax.numpy as jnp

import numpy as np

# ASSUMPTIONS
# Using a marginalizing timing model
# Plain noise model - uses pre-v1 caching
# Only computes the noise-marginalized per-frequency pair-covariant OS - no maximum likelihood, no multiple component, no non-pair-covariant

# hasn't been tested extremely thoroughly but I computed a few draws and they pass np.allclose with atol=0 when compared to official defiant
# (apparently the default atol is not suitable for comparing very small numbers as described in the NumPy documentation, so I set it to 0)
# seems to give a factor of a few times speedup on CPU but can now run on GPU for a big speedup obviously

# Because I don't feel like making a notebook tutorial, here is an example usage
# import jax.numpy as jnp
# from defiant import nmpfpcos_jax as nj
# <load your pulsars and pta and lfcore here>
# npsrs, FNdt, TNdt, TNT, FNT, FNF, pair_idx, xi = nj.cpu_cache(psrs, pta)
# N = 20 # number of noise draws
# frequencies = jnp.array([1,3,7]) # or whatever frequencies you want. 
# could be e.g. jnp.array([4]) for single-frequency or jnp.arange for the whole spectrum
# phi, phiinv = nj.cpu_prep(npsrs, pta, lfcore, N, frequencies)
# rhok, sigk, Sk, Ck = nj.gpu_nmpfpcos(FNdt, TNdt, TNT, FNT, FNF, phiinv, pair_idx, xi, phi, frequencies) # this part uses the GPU if there is one 

# run this once first, can use the results for any frequency and noise draw
def cpu_cache(psrs, pta):
    npsrs = len(psrs)
    pair_idx = np.array(np.triu_indices(npsrs,1)).T
    xi = np.arccos([np.dot(psrs[a].pos, psrs[b].pos) for (a,b) in pair_idx])
    def xNy(N, x, y):
        term1 = N.Nmat.solve(y, left_array=x)
        term2 = N.MNF(x).T @ N.MNMMNF(y)
        return term1 - term2
    
    FNdt = []
    TNdt = []
    TNT = []
    FNT = []
    FNF = []
    for i in range(npsrs):
        
        F = pta._signalcollections[i]['gw'].get_basis()
        N = pta._signalcollections[i].get_ndiag()
        T = pta._signalcollections[i].get_basis()
        dt = pta._signalcollections[i].get_detres()
        
        FNdt.append(xNy(N,F,dt))
        TNdt.append(xNy(N,T,dt))
        TNT.append(xNy(N,T,T))
        FNT.append(xNy(N,F,T))
        FNF.append(xNy(N,F,F))
        
    return npsrs, jnp.array(FNdt), jnp.array(TNdt), jnp.array(TNT), jnp.array(FNT), jnp.array(FNF), jnp.array(pair_idx), jnp.array(xi)

# this one depends on noise and frequencies - uses the CPU to get the enterprise part out of the way
def cpu_prep(npsrs, pta, lfcore, N, frequencies):
    gw_sig = [s for s in pta._signalcollections[0].signals if s.signal_id=='gw'][0]

    phi = []
    phiinv = []
    for noise_draw in range(N):
        _phiinv = []
        idx = np.random.randint(lfcore.burn, lfcore.chain.shape[0])
        pars = dict(zip(lfcore.params, lfcore.chain[idx]))
        for psr_idx in range(npsrs):
            __phiinv = pta._signalcollections[psr_idx].get_phiinv(pars)
            if __phiinv.ndim == 1:
                __phiinv = np.diag(__phiinv)
            _phiinv.append(__phiinv)
        phiinv.append(_phiinv)
        phi.append(gw_sig.get_phi(pars))

    return jnp.array(phi), jnp.array(phiinv)

# this runs on GPU or CPU; pass it the outputs of the previous two functions
@jax.jit
def gpu_nmpfpcos(FNdt, TNdt, TNT, FNT, FNF, phiinv, pair_idx, xi, phi, frequencies):
    X, Z = jax.vmap(jax.vmap(compute_XZ, in_axes=(0,0,0,0,0,0)), in_axes=(None,None,None,None,None,0))(FNdt, TNdt, TNT, FNT, FNF, phiinv) # vmap over noise draws and pulsars
    # orf matrix and orf design matrix
    def hd(xi): # only valid for cross-correlations; could do jnp.where(xi==0, 1, .5 - stuff) if want a general HD function
        x = (1-jnp.cos(xi)) / 2
        return 0.5 - x * ( 0.25 - 1.5*jnp.log(x) )
    hd_of_xi = hd(xi)
    a,b = pair_idx[:,0], pair_idx[:,1]
    orf_matrix = jnp.ones((X.shape[1],X.shape[1])).at[a,b].set(hd_of_xi).at[b,a].set(hd_of_xi)
    orf_design_matrix = hd_of_xi

    phi_til = jnp.zeros(X.shape[2]//2)

    a,b = pair_idx[:,0], pair_idx[:,1]
    def ZphiZphihat_n(n): # ZphiZphihat for a single noise-draw n
        ZphiZphihat = jnp.zeros((Z.shape[1],Z.shape[1], Z.shape[2],Z.shape[2]))
        ZphiZphihat = ZphiZphihat.at[a,b].set(((phi[n]*Z[n,a]) @ Z[n,b]))
        ZphiZphihat = ZphiZphihat.at[b,a].set(((phi[n]*Z[n,b]) @ Z[n,a]))
        return ZphiZphihat
    ZphiZphihat = jax.vmap(ZphiZphihat_n)(jnp.arange(Z.shape[0])) # vmap over noise draws

    npair = Z.shape[1]*(Z.shape[1]-1)//2
    PoP_idx = jnp.array(jnp.triu_indices(npair)).T
    
    PoP = jnp.zeros((len(PoP_idx),4),dtype=int)
    PoP = PoP.at[:,(0,1)].set(pair_idx[PoP_idx[:,0]])
    PoP = PoP.at[:,(2,3)].set(pair_idx[PoP_idx[:,1]])

    # It is also helpful to create some basic filters. (Kyle)
    psr_match = (PoP[:,(0,1)] == PoP[:,(2,3)]) # checks (a==c,b==d) (Kyle)
    psr_inv_match = (PoP[:,(0,1)] == PoP[:,(3,2)]) # checks (a==d,b==c) (Kyle)

    conditions = [psr_match[:,0] & psr_match[:,1],
                  psr_match[:,0] & ~psr_match[:,1],
                  ~psr_inv_match[:,0] & psr_inv_match[:,1],
                  ~psr_match[:,0] & psr_match[:,1]]
    
    p_idx1, p_idx2 = PoP_idx.T
    a_PoP,b_PoP,c_PoP,d_PoP = PoP.T

    # a single-frequency PCOS
    def pcos(X, Z, ZphiZphihat, orf_matrix, phi, k):
        rho, sig, norm = compute_rhok_sigk(a, b, phi_til, X, Z, phi, k)
        s_diag = jnp.diag(sig**2)
        # select the 2x2 frequency patch -- seems to save memory
        Zphihat = jnp.zeros((Z.shape[0],2,2)).at[:,0,0].set(Z[:,2*k,2*k]).at[:,0,1].set(Z[:,2*k,2*k+1]).at[:,1,0].set(Z[:,2*k+1,2*k]).at[:,1,1].set(Z[:,2*k+1,2*k+1])
        ZphiZphihat = jnp.zeros((Z.shape[0],Z.shape[0],2,2)).at[:,:,0,0].set(ZphiZphihat[:,:,2*k,2*k]).at[:,:,0,1].set(ZphiZphihat[:,:,2*k,2*k+1]).at[:,:,1,0].set(ZphiZphihat[:,:,2*k+1,2*k]).at[:,:,1,1].set(ZphiZphihat[:,:,2*k+1,2*k+1])
        C = create_PFOS_pair_covariance(Zphihat, ZphiZphihat, orf_matrix, norm, conditions, p_idx1,p_idx2, a_PoP,b_PoP,c_PoP,d_PoP)
        S = woodbury_linear_solve(orf_design_matrix, C, rho, s_diag)
        return rho, sig, S, C
    
    pfpcos = jax.vmap(pcos, in_axes=(None,None,None,None,None,0)) # vmap over frequencies
    nmpfpcos = jax.vmap(pfpcos, in_axes=(0,0,0,None,0,None)) # vmap over noise draws
    rhok, sigk, Sk, Ck = nmpfpcos(X, Z, ZphiZphihat, orf_matrix, phi, frequencies)
    return rhok, sigk, Sk[:,:,0,0], Ck

def compute_XZ(FNdt, TNdt, TNT, FNT, FNF, phiinv):
    sigma = phiinv + TNT
    X = FNdt - FNT @ jnp.linalg.solve(sigma, TNdt)
    Z = FNF - FNT @ jnp.linalg.solve(sigma, FNT.T)
    return X, Z

def compute_rhok_sigk(a, b, phi_til, X, Z, phi, freq):

    # Compute rho_ab(f_{freq}), sigma_ab(f_{freq}), and normalization_ab(f_{freq}) (Kyle)

    phi_til = phi_til.at[freq].set(1)
    phi_til = jnp.repeat(phi_til,2)
    phi2 = phi/phi[2*freq]

    norms_abk = 1/(jnp.einsum('ijk,ikj->i',phi_til*Z[a],phi2*Z[b]))

    rho_abk =  jnp.sum(X[a] * phi_til * X[b], axis=1) * norms_abk
    sig_abk =  jnp.sqrt(jnp.einsum('ijk,ikj->i', phi_til*Z[a], phi_til*Z[b]) * norms_abk**2)

    return rho_abk, sig_abk, norms_abk

def woodbury_linear_solve(X, C, r, s):
    X = X[:,None] # assumes X is 1D
    r = r[:,None] # assumes r is 1D

    A = s
    K = C - A
    In = jnp.eye(A.shape[0])

    cinv = woodbury_inverse(A,In,In,K)

    fisher = X.T @ cinv @ X
    dirty_map = X.T @ cinv @ r 

    cov = jnp.linalg.pinv(fisher)
        
    theta = cov.T @ dirty_map
    return theta # does not return uncertainty on Sk because I only use point estimates for now

def woodbury_inverse(A, U, C, V): 
    # basically from stack overflow 
    # assumes C is the identity
    # has benefit of vector multiplication instead of diagonal matrix multiplication
    A_inv_diag = 1/jnp.diag(A) 
    B_inv = jnp.linalg.inv(C + (V * A_inv_diag) @ U)
    return jnp.diag(A_inv_diag) - (A_inv_diag.reshape(-1,1) * U @ B_inv @ V * A_inv_diag)

def create_PFOS_pair_covariance(Zphihat, ZphiZphihat, orf, norm_ab, conditions, p_idx1,p_idx2, a,b,c,d):

    # Define the three cases for the pair covariance (Kyle - the rest of the comments in this function)
    def case1(a,b,c,d):             #(ab,cd)
        # = gamma_{ac} gamma_{bd} tr([Z_d phi Z_b] phihat [Z_a phi Z_c] phihat) + 
        #   gamma_{ad} gamma_{bc} tr([Z_c phi Z_b] phihat [Z_a phi Z_d] phihat)
        a4 = orf[a,c]*orf[d,b] * mpt(ZphiZphihat[d,b], ZphiZphihat[a,c]) + \
             orf[a,d]*orf[c,b] * mpt(ZphiZphihat[c,b], ZphiZphihat[a,d])
        return a4
    
    def case2(a,b,c):               #(ab,ac)
        # = gamma_{bc}            tr([Z_c phi Z_b] phihat [Z_a] phihat) + 
        #   gamma_{ac} gamma_{ab} tr([Z_a phi Z_b] phihat [Z_a phi Z_c] phihat)
        a2 = orf[b,c] *          mpt(ZphiZphihat[c,b],Zphihat[a])
        a4 = orf[a,c]*orf[a,b] * mpt(ZphiZphihat[a,b],ZphiZphihat[a,c])
        return a2+a4

    def case3(a,b):                 #(ab,ab)
        # = tr(Z_b phihat Z_a phihat) + gamma_{ab}^2 tr([Z_a phi Z_b] phihat [Z_a phi Z_b] phihat)
        a0 =               mpt(Zphihat[b],Zphihat[a])
        a4 = orf[a,b]**2 * mpt(ZphiZphihat[b,a],ZphiZphihat[b,a])
        return a0+a4

    choices = [case3(a,b), case2(a,b,d), case2(b,a,d), case2(b,a,c)]
    
    C_m = jnp.select(conditions, choices, default=case1(a,b,c,d))
    C_m = jnp.zeros((norm_ab.shape[0],norm_ab.shape[0])).at[p_idx1,p_idx2].set(C_m).at[p_idx2,p_idx1].set(C_m)

    # Include the final sigmas
    C_m *= jnp.outer(norm_ab,norm_ab)

    return C_m

def mpt(A,B): # matrix product trace -- assumes 3D A and B
    return jnp.einsum('ijk,ikj->i',A,B)
