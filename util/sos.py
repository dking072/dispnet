from pyscf import gto, scf, dft, tdscf
import numpy as np
import matplotlib.pyplot as plt

spin_dct = {
    "h":1,
    "h2":0,
    "c":2,
    "ch":1,
    "c2h2":0,
    "c2h4":0,
    "c2h6":0,
    "o":2,
    "oh":1,
    "oh2":0,
    "n":3,
    "nh":2,
    "n2h2":0,
    "nh3":0,
    "f":1,
    "fh":0,
    "p":3,
    "ph":2,
    "p2h2":0,
    "ph3":0,
    "s":2,
    "sh":1,
    "sh2":0,
    "cl":1,
    "clh":0,
    "br":1,
    "brh":0,
    "i":1,
    "ih":0,
}

def calc_alpha_scalar(zeta,e,dyadic_tr,nstates=-1): #vectorized
    num = 2*e*dyadic_tr #(N,C)
    denom = e[:,:,None]**2 + zeta[None,None,:]**2 #(N,C,G)
    alpha_c = (num[:,:,None] / denom) #(N,C,G)
    idx = np.argsort(alpha_c[0,:,0])[::-1][:nstates]
    alpha = alpha_c[:,idx,:].sum(axis=1) #(N,G)
    alpha_c = alpha_c[:,:,0]
    idx = np.where(alpha_c > 0.1)[-1]
    # print("Greater than 0.1 at zeta=0:",len(idx))
    # print("Their energies:",e[:,idx].mean(),"+/-",e[:,idx].std())
    return alpha, alpha_c

def calc_alpha_matrix(zeta,e,mu,nstates=-1):
    dyadic = mu[:,:,:,None] * mu[:,:,None,:] #(N,C,3,1)(N,C,1,3)->(N,C,3,3)
    num = 2*e[:,:,None,None]*dyadic #(N,C,3,3)
    denom = e[:,:,None,None,None]**2 + zeta[None,None,None,None,:]**2 #(N,C,3,3,G)
    alpha = (num[...,None] / denom) #(N,C,3,3,G)
    alpha_tr = alpha.diagonal(axis1=-3,axis2=-2).sum(axis=-1)/3 #(N,C,G)
    idx = np.argsort(alpha_tr[0,:,0])[::-1][:nstates]
    alpha = alpha[:,idx,...].sum(axis=1) #(N,3,3,G)
    alpha_tr = alpha_tr[:,idx,...].sum(axis=1) #(N,G)
    # print("Selected state energies:",e[:,idx].mean(),e[:,idx].std())
    return alpha_tr, alpha

def ase_to_pyscf_mol(atm, basis="def2qzvppd", unit='Angstrom'):
    m = atm.info["m"]
    mol = gto.Mole()
    mol.unit = unit
    mol.atom = [(atom.symbol, atom.position) for atom in atm]
    mol.basis = basis
    mol.spin = spin_dct[m]
    mol.verbose = 0
    try:
        mol.build()
    except RuntimeError:
        mol.spin = 1
        mol.build()
    return mol
    
def calc_c6(atm,basis="def2qzvppd",nstates=-1,nalpha=-1,plot=False):
    mol = ase_to_pyscf_mol(atm)
    
    nmo, nocc = mol.nao, mol.nelec[0]
    # print("Number of basis functions:",mol.nao)
    nvir = nmo - nocc

    if mol.spin != 0:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = '.375*hf + .625*pbe,pbe'
    mf.kernel()
    
    mf_td = tdscf.TDDFT(mf)
    mf_td.nstates = nocc * nvir
    # print("Tot states:",nocc * nvir)
    mf_td.kernel()
    td_transdip = mf_td.transition_dipole()
    td_eig = mf_td.e #in Ha
    td_eig = td_eig[:nstates]
    td_transdip = td_transdip[:nstates]
    
    #Grid the imaginary freqs for integration
    n_grid = 200
    zeta = np.linspace(0, 10, n_grid)

    #Calculate alpha
    e = td_eig[None,:]
    mu = td_transdip[None,:]
    dyadic = mu[:,:,:,None] * mu[:,:,None,:] #(N,C,3,1)(N,C,1,3)->(N,C,3,3)
    dyadic_avg = dyadic.diagonal(axis1=-2,axis2=-1).sum(axis=-1) / 3 #(N,C)
    #DYADIC TR IS AVGD OVER 3 DIMENSTIONS

    #Calculate in 2 different ways
    #ALPHA IS ALPHA AVGD OVER THE 3 DIMENSIONS (Tr(alpha)/3)
    alpha1, alpha_matrix = calc_alpha_matrix(zeta,e,mu,nstates=nalpha)
    alpha2, alpha_c = calc_alpha_scalar(zeta,e,dyadic_avg,nstates=nalpha)
    if (nstates == -1) and (nalpha == -1):
        assert(np.allclose(alpha1,alpha2))
    alpha = alpha1

    #Calculate C6
    y = alpha**2
    c6 = ((3 / np.pi) * np.trapz(y,zeta))[0]
    # print(f"{atm}-{atm} c6: {c6:1f}")
    c6 = np.round(c6,2)

    #Plot alpha
    if plot:
        plt.plot(zeta,alpha[0])
        plt.title(f"Plot of {atm} $\\alpha$",fontsize=15)
        plt.annotate(f"$C_6$ = {c6}",(6,1),fontsize=15)
        plt.xlabel("$\zeta$ (Ha)",fontsize=15)
        plt.ylabel("$\\alpha$",fontsize=15)

    out = {
        "alpha":alpha_matrix,
        "alpha_avg":alpha,
        "c6":c6,
        "zeta":zeta,
        "e":e.ravel(),
        "mu":mu,
        "dyad_avg":dyadic_avg.ravel(),
    }
    return out