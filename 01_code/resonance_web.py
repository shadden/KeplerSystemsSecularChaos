import numpy as np
import matplotlib.pyplot as plt
import rebound as rb
import celmech as cm
import sympy as sp
from three_body_mmr import *
from celmech.poisson_series import bracket, PSTerm
import sys

J = int(sys.argv[1])
m = 3.e-6
ex_frac = 0.25

#MMRs = [(J,1),(3*J+1,3),(2*J+1,2),(3*J+2,3),(J+1,1)] 
MMRs = [(J,1),(2*J+1,2),(J+1,1)] 

def get_rebound_sim(ms,Ps,es,ls,pmgs):
    sim = rb.Simulation()
    sim.add(m=1)
    for m,P,e,l,pmg in zip(ms,Ps,es,ls,pmgs):
        sim.add(m=m,P=P,e=e,l=l,pomega=pmg)
    sim.move_to_com()
    return sim

inner_Pjk,outer_Pjk = {},{}
inner_sx_p,outer_sx_p = {},{}
inner_sx_m,outer_sx_m = {},{}
Nsample = 600
NDelta = 30
pmgs = np.random.uniform(-np.pi,np.pi,(Nsample,3))
Z = np.exp(1j * pmgs)

for j,k in MMRs:
    P = j/(j-k)
    Ps=P**np.arange(3)
    ms = m * np.ones(3)
    alpha = P**(-2/3)
    ex = (P**(2/3)-1)/((P**(2/3)+1)) #(1-alpha) * np.ones(3)
    eccs = ex_frac * ex * np.ones(3)
    ls,pomegas = np.zeros((2,3))
    sim = get_rebound_sim(ms,Ps,eccs,ls,pomegas)
    pvars = cm.Poincare.from_Simulation(sim)
    pham = cm.PoincareHamiltonian(pvars)
    terms = list_resonance_terms(j,k,inclinations=False)
    inner_terms = [DFTerm_as_PSterms(pham,1,2,kvec,nu_vec,(0,0))[0] for kvec,nu_vec in terms]
    outer_terms = [DFTerm_as_PSterms(pham,2,3,kvec,nu_vec,(0,0))[0] for kvec,nu_vec in terms]
    # reduce to poisson series in xs only
    inner_terms = [PSTerm(term.C,term.k[:3],term.kbar[:3],[],[]) for term in inner_terms]
    outer_terms = [PSTerm(term.C,term.k[:3],term.kbar[:3],[],[]) for term in outer_terms] 
    Pjk_in = PoissonSeries.from_PSTerms(inner_terms)
    Pjk_out = PoissonSeries.from_PSTerms(outer_terms)
    
    inner_Pjk[(j,k)] = Pjk_in
    outer_Pjk[(j,k)] = Pjk_out

    # width determinations
    omega = pham.flow_func(*pham.state.values)[:pham.N_dof:3].reshape(-1)
    domega = np.diag(pham.jacobian_func(*pham.state.values)[:pham.N_dof:3,pham.N_dof::3])
    Lmbda0s = [pham.H_params[X] for X in pham.Lambda0s[1:]]
    jvec_in = np.array((k-j,j,0))
    jvec_out = np.array((0,k-j,j))
    Minv_in = np.sum(jvec_in**2 * domega)
    Minv_out = np.sum(jvec_out**2 * domega)

    xs = np.sqrt(Lmbda0s) * eccs * Z/np.sqrt(2)
    dIs_inner = np.array([2 * np.sqrt(2 * np.abs(Pjk_in(x,[],[]) / Minv_in)) for x in xs])
    dIs_outer = np.array([2 * np.sqrt(2 * np.abs(Pjk_out(x,[],[]) / Minv_out)) for x in xs])
    inner_sx_p[(j,k)] = [omega + domega * jvec_in  * np.quantile(dIs_inner,q) for q in [0.5,1]]
    outer_sx_p[(j,k)] = [omega + domega * jvec_out * np.quantile(dIs_outer,q) for q in [0.5,1]]
    inner_sx_m[(j,k)] = [omega - domega * jvec_in  * np.quantile(dIs_inner,q) for q in [0.5,1]]
    outer_sx_m[(j,k)] = [omega - domega * jvec_out * np.quantile(dIs_outer,q) for q in [0.5,1]]


n_ext = {}
for j1,k1 in MMRs:
    Pjk_in = inner_Pjk[(j1,k1)]
    if j1==J:
        continue
    pr = inner_sx_p[(j1,k1)][0][0]/inner_sx_p[(j1,k1)][0][1]
    Delta_min = ((j1-k1)/j1) * pr - 1
    Delta_max = ((j1-k1)/j1) * (J/(J-1)) - 1
    Deltas = np.linspace(Delta_min,Delta_max,NDelta)
    for j2,k2 in MMRs:
        if j2==J:
            continue
        jvec = np.array([j1-k1,-j1-j2+k2,j2])
        Pjk_out = outer_Pjk[(j2,k2)]
        term1 = bracket(Pjk_in,1j * Pjk_out.conj)
        term2 = Pjk_in * Pjk_out.conj
        n_ext_p_50,n_ext_m_50,n_ext_p_max,n_ext_m_max = np.zeros((4,len(Deltas),3))
        for i,Delta in enumerate(Deltas):
            n,dn,Lambda = three_body_mmr_n_and_dn(j1,k1,j2,k2,-1,ms,Delta,n1=2*np.pi,GM=1)
            smas = (sim.G/n**2)**(1/3)
            P2ex = lambda P: (P**(2/3)-1)/(P**(2/3)+1)
            exs = np.array([P2ex(n[0]/n[1]),P2ex(n[1]/n[2])])
            eccs = ex_frac * np.array([0.5 * exs[0], 0.5 * (0.5 * exs[0] + 0.5 * exs[1]) , 0.5 * exs[1]])
            xs = np.sqrt(ms * np.sqrt(sim.G * smas)/2) * eccs * Z
            C1,C2 = Q_factors(j1,k1,j2,k2,n,dn)
            Qs =  np.array([C1 * term1(x,[],[]) + C2 * term2(x,[],[]) for x in xs])
            Minv = jvec**2 @ dn
            dI_50 = np.quantile(2 * np.sqrt(np.abs(Qs / Minv)),0.5)
            dI_max = np.quantile(2 * np.sqrt(np.abs(Qs / Minv)),1)
            n_ext_p_50[i] = n + jvec * dn * dI_50 
            n_ext_m_50[i] = n - jvec * dn * dI_50 
            n_ext_p_max[i] = n + jvec * dn * dI_max 
            n_ext_m_max[i] = n - jvec * dn * dI_max
        n_ext[(j1,k1,j2,k2)] = (n_ext_p_50,n_ext_m_50,n_ext_p_max,n_ext_m_max)


n_ext2 = {}
for j1,k1 in MMRs:
    Pjk_in = inner_Pjk[(j1,k1)]
    if j1==J+1:
        continue
    pr = inner_sx_m[(j1,k1)][0][0]/inner_sx_m[(j1,k1)][0][1]
    Delta_min = ((j1-k1)/j1) * pr - 1
    Delta_max = ((j1-k1)/j1) * ((J+1)/J) - 1
    Deltas = np.linspace(Delta_min,Delta_max,NDelta)
    for j2,k2 in MMRs:
        if j2==J+1:
            continue
        jvec = np.array([j1-k1,-j1-j2+k2,j2])
        Pjk_out = outer_Pjk[(j2,k2)]
        term1 = bracket(Pjk_in,1j * Pjk_out.conj)
        term2 = Pjk_in * Pjk_out.conj
        n_ext_p_50,n_ext_m_50,n_ext_p_max,n_ext_m_max = np.zeros((4,len(Deltas),3))
        for i,Delta in enumerate(Deltas):
            n,dn,Lambda = three_body_mmr_n_and_dn(j1,k1,j2,k2,-1,ms,Delta,n1=2*np.pi,GM=1)
            smas = (sim.G/n**2)**(1/3)
            P2ex = lambda P: (P**(2/3)-1)/(P**(2/3)+1)
            exs = np.array([P2ex(n[0]/n[1]),P2ex(n[1]/n[2])])
            eccs = ex_frac * np.array([0.5 * exs[0], 0.5 * (0.5 * exs[0] + 0.5 * exs[1]) , 0.5 * exs[1]])
            xs = np.sqrt(ms * np.sqrt(sim.G * smas)/2) * eccs * Z
            C1,C2 = Q_factors(j1,k1,j2,k2,n,dn)
            Qs =  np.array([C1 * term1(x,[],[]) + C2 * term2(x,[],[]) for x in xs])
            Minv = jvec**2 @ dn
            dI_50 = np.quantile(2 * np.sqrt(np.abs(Qs / Minv)),0.5)
            dI_max = np.quantile(2 * np.sqrt(np.abs(Qs / Minv)),1)
            n_ext_p_50[i] = n + jvec * dn * dI_50 
            n_ext_m_50[i] = n - jvec * dn * dI_50 
            n_ext_p_max[i] = n + jvec * dn * dI_max 
            n_ext_m_max[i] = n - jvec * dn * dI_max
        n_ext2[(j1,k1,j2,k2)] = (n_ext_p_50,n_ext_m_50,n_ext_p_max,n_ext_m_max)

#### Plus
n_ext_plus = {}

j1,k1 = J,1
j2,k2 = J+1,1
jvec = np.array([k1-j1,j1,0]) + np.array([0,k2 - j2,j2])
Pjk_in = inner_Pjk[(j1,k1)]
Pjk_out = inner_Pjk[(j2,k2)]
term = Pjk_in * Pjk_out * j1 * (j2-k2)
pr = inner_sx_m[(j1,k1)][0][0]/inner_sx_m[(j1,k1)][0][1]
Delta_min = ((j1-k1)/j1) * pr - 1
Delta_max = ((j1-k1)/j1) * ((J+1)/J) - 1
Deltas = np.linspace(Delta_min,Delta_max,NDelta)
n_ext_p_50,n_ext_m_50,n_ext_p_max,n_ext_m_max = np.zeros((4,len(Deltas),3))
for i,Delta in enumerate(Deltas):
    n,dn,Lambda = three_body_mmr_n_and_dn(j1,k1,j2,k2,+1,ms,Delta,n1=2*np.pi,GM=1)
    smas = (sim.G/n**2)**(1/3)
    P2ex = lambda P: (P**(2/3)-1)/(P**(2/3)+1)
    exs = np.array([P2ex(n[0]/n[1]),P2ex(n[1]/n[2])])
    eccs = ex_frac * np.array([0.5 * exs[0], 0.5 * (0.5 * exs[0] + 0.5 * exs[1]) , 0.5 * exs[1]])
    xs = np.sqrt(ms * np.sqrt(sim.G * smas)/2) * eccs * Z
    omega_in = np.array([k1-j1,j1,0]) @ n
    omega_out = np.array([0,k2 - j2,j2])@ n
    C = dn[1] * (omega_out**2 + omega_in**2)/(omega_out**2 * omega_in**2)
    Qs = np.array([C * term(x,[],[]) for x in xs])
    Minv = jvec**2 @ dn
    dI_50 = np.quantile(2 * np.sqrt(np.abs(Qs / Minv)),0.5)
    dI_max = np.quantile(2 * np.sqrt(np.abs(Qs / Minv)),1)
    n_ext_p_50[i] = n + jvec * dn * dI_50 
    n_ext_m_50[i] = n - jvec * dn * dI_50 
    n_ext_p_max[i] = n + jvec * dn * dI_max 
    n_ext_m_max[i] = n - jvec * dn * dI_max
n_ext_plus[(j1,k1,j2,k2)] = (n_ext_p_50,n_ext_m_50,n_ext_p_max,n_ext_m_max)


j1,k1 = J+1,1
j2,k2 = J,1
jvec = np.array([k1-j1,j1,0]) + np.array([0,k2 - j2,j2])
Pjk_in = inner_Pjk[(j1,k1)]
Pjk_out = inner_Pjk[(j2,k2)]
term = Pjk_in * Pjk_out * j1 * (j2-k2)
pr = inner_sx_p[(j1,k1)][0][0]/inner_sx_p[(j1,k1)][0][1]
Delta_min = ((j1-k1)/j1) * pr - 1
Delta_max = ((j1-k1)/j1) * ((J)/(J-1)) - 1
Deltas = np.linspace(Delta_min,Delta_max,NDelta)
n_ext_p_50,n_ext_m_50,n_ext_p_max,n_ext_m_max = np.zeros((4,len(Deltas),3))

for i,Delta in enumerate(Deltas):
    n,dn,Lambda = three_body_mmr_n_and_dn(j1,k1,j2,k2,+1,ms,Delta,n1=2*np.pi,GM=1)
    smas = (sim.G/n**2)**(1/3)
    P2ex = lambda P: (P**(2/3)-1)/(P**(2/3)+1)
    exs = np.array([P2ex(n[0]/n[1]),P2ex(n[1]/n[2])])
    eccs = ex_frac * np.array([0.5 * exs[0], 0.5 * (0.5 * exs[0] + 0.5 * exs[1]) , 0.5 * exs[1]])
    xs = np.sqrt(ms * np.sqrt(sim.G * smas)/2) * eccs * Z
    omega_in = np.array([k1-j1,j1,0]) @ n
    omega_out = np.array([0,k2 - j2,j2])@ n
    C = dn[1] * (omega_out**2 + omega_in**2)/(omega_out**2 * omega_in**2)
    Qs = np.array([C * term(x,[],[]) for x in xs])
    Minv = jvec**2 @ dn
    dI_50 = np.quantile(2 * np.sqrt(np.abs(Qs / Minv)),0.5)
    dI_max = np.quantile(2 * np.sqrt(np.abs(Qs / Minv)),1)
    n_ext_p_50[i] = n + jvec * dn * dI_50 
    n_ext_m_50[i] = n - jvec * dn * dI_50 
    n_ext_p_max[i] = n + jvec * dn * dI_max 
    n_ext_m_max[i] = n - jvec * dn * dI_max
n_ext_plus[(j1,k1,j2,k2)] = (n_ext_p_50,n_ext_m_50,n_ext_p_max,n_ext_m_max)


j1,k1 = J,1
j2,k2 = 2*J+1,2
jvec = np.array([k1-j1,j1,0]) + np.array([0,k2 - j2,j2])
Pjk_in = inner_Pjk[(j1,k1)]
Pjk_out = inner_Pjk[(j2,k2)]
term = Pjk_in * Pjk_out * j1 * (j2-k2)
pr = inner_sx_m[(j1,k1)][0][0]/inner_sx_m[(j1,k1)][0][1]
Delta_min = ((j1-k1)/j1) * pr - 1
Delta_max = ((j1-k1)/j1) * ((J+1)/J) - 1
Deltas = np.linspace(Delta_min,Delta_max,NDelta)
n_ext_p_50,n_ext_m_50,n_ext_p_max,n_ext_m_max = np.zeros((4,len(Deltas),3))
for i,Delta in enumerate(Deltas):
    n,dn,Lambda = three_body_mmr_n_and_dn(j1,k1,j2,k2,+1,ms,Delta,n1=2*np.pi,GM=1)
    smas = (sim.G/n**2)**(1/3)
    P2ex = lambda P: (P**(2/3)-1)/(P**(2/3)+1)
    exs = np.array([P2ex(n[0]/n[1]),P2ex(n[1]/n[2])])
    eccs = ex_frac * np.array([0.5 * exs[0], 0.5 * (0.5 * exs[0] + 0.5 * exs[1]) , 0.5 * exs[1]])
    xs = np.sqrt(ms * np.sqrt(sim.G * smas)/2) * eccs * Z
    omega_in = np.array([k1-j1,j1,0]) @ n
    omega_out = np.array([0,k2 - j2,j2])@ n
    C = dn[1] * (omega_out**2 + omega_in**2)/(omega_out**2 * omega_in**2)
    Qs = np.array([C * term(x,[],[]) for x in xs])
    Minv = jvec**2 @ dn
    dI_50 = np.quantile(2 * np.sqrt(np.abs(Qs / Minv)),0.5)
    dI_max = np.quantile(2 * np.sqrt(np.abs(Qs / Minv)),1)
    n_ext_p_50[i] = n + jvec * dn * dI_50 
    n_ext_m_50[i] = n - jvec * dn * dI_50 
    n_ext_p_max[i] = n + jvec * dn * dI_max 
    n_ext_m_max[i] = n - jvec * dn * dI_max
n_ext_plus[(j1,k1,j2,k2)] = (n_ext_p_50,n_ext_m_50,n_ext_p_max,n_ext_m_max)


j1,k1 = J+1,1
j2,k2 = 2*J+1,2
jvec = np.array([k1-j1,j1,0]) + np.array([0,k2 - j2,j2])
Pjk_in = inner_Pjk[(j1,k1)]
Pjk_out = inner_Pjk[(j2,k2)]
term = Pjk_in * Pjk_out * j1 * (j2-k2)
pr = inner_sx_p[(j1,k1)][0][0]/inner_sx_p[(j1,k1)][0][1]
Delta_min = ((j1-k1)/j1) * pr - 1
Delta_max = ((j1-k1)/j1) * ((J)/(J-1)) - 1
Deltas = np.linspace(Delta_min,Delta_max,NDelta)
n_ext_p_50,n_ext_m_50,n_ext_p_max,n_ext_m_max = np.zeros((4,len(Deltas),3))

for i,Delta in enumerate(Deltas):
    n,dn,Lambda = three_body_mmr_n_and_dn(j1,k1,j2,k2,+1,ms,Delta,n1=2*np.pi,GM=1)
    smas = (sim.G/n**2)**(1/3)
    P2ex = lambda P: (P**(2/3)-1)/(P**(2/3)+1)
    exs = np.array([P2ex(n[0]/n[1]),P2ex(n[1]/n[2])])
    eccs = ex_frac * np.array([0.5 * exs[0], 0.5 * (0.5 * exs[0] + 0.5 * exs[1]) , 0.5 * exs[1]])
    xs = np.sqrt(ms * np.sqrt(sim.G * smas)/2) * eccs * Z
    omega_in = np.array([k1-j1,j1,0]) @ n
    omega_out = np.array([0,k2 - j2,j2])@ n
    C = dn[1] * (omega_out**2 + omega_in**2)/(omega_out**2 * omega_in**2)
    Qs = np.array([C * term(x,[],[]) for x in xs])
    Minv = jvec**2 @ dn
    dI_50 = np.quantile(2 * np.sqrt(np.abs(Qs / Minv)),0.5)
    dI_max = np.quantile(2 * np.sqrt(np.abs(Qs / Minv)),1)
    n_ext_p_50[i] = n + jvec * dn * dI_50 
    n_ext_m_50[i] = n - jvec * dn * dI_50 
    n_ext_p_max[i] = n + jvec * dn * dI_max 
    n_ext_m_max[i] = n - jvec * dn * dI_max
n_ext_plus[(j1,k1,j2,k2)] = (n_ext_p_50,n_ext_m_50,n_ext_p_max,n_ext_m_max)

import pickle
with open("resonance_web_data_J{:d}.pkl".format(J),"wb") as fi:
    save = {
        'n_ext':n_ext,
        'n_ext2':n_ext2,
        'n_ext_plus':n_ext_plus,
        "inner_sx_m":inner_sx_m,
        "inner_sx_p":inner_sx_p,
        "outer_sx_m":outer_sx_m,        
        "outer_sx_p":outer_sx_p,
        "inner_Pjk":inner_Pjk,
        "outer_Pjk":outer_Pjk
        }
    pickle.dump(save,fi)