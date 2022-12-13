#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:09:38 2022

@author: sergej

Markov game with options "forage" and "lurk"
"""
import numpy as np
import copy
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# =============================================================================
# Item selection - utility = prisoner's dilemma
# =============================================================================
# Compute expected gain based on transitions
def comp_EG(pG, b, c):
    pE = (1-pG)/2
    r = (pG+pE)*(b/2)+(1-(pG+pE))*c
    s = c
    t = pG*b+(1-pG)*d
    p = d
    return r, s, t, p, pE

# Variables
pRaw = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
# Constants
b = 2
c = -2
d = -1

# All item configurations - single weather
items = []
for i in range(0, len(pRaw)):
            # weather p and magnitude combinations
            pG = pRaw[i]
            # Append parameters and and expected payoffs
            tabl = {"1 pSuccess":[], "1.1 pExtra coop":[], "2 gain magnitude":[],
                    "3 r":[], "4 s":[], "5 t":[], "6 p":[]}
            r, s, t, p, pE = comp_EG(pG, b, c)
            tabl["1 pSuccess"].append(pG)
            tabl["1.1 pExtra coop"].append(pE)
            tabl["2 gain magnitude"].append(b)
            tabl["3 r"].append(r)
            tabl["4 s"].append(s)
            tabl["5 t"].append(t)
            tabl["6 p"].append(p)
            items.append(tabl)
# Select items referring to prisoner's dilemma
slct = []
for i in range(0, len(items)):
    if items[i]["5 t"] > items[i]["3 r"] > items[i]["6 p"] > items[i]["4 s"]:
        slct.append(items[i])
# Number of environment states
nEnv = len(slct)
## Export items
# slctR = pd.DataFrame(slct)
# slctR.to_csv('/home/sergej/Documents/academics/dnhi/projects/prisoner_forests.csv', index = False)
# =============================================================================
# # Select all items even if not prisoner's dil.
# slct = items
# =============================================================================

# =============================================================================
# Markov Decision Process
# =============================================================================
## Transitions
# Choice options & transitions
def foraL(trns_vec, p_trans, lpCur):  # Subject waits
    trns_vec = transitio(lpCur, c, trns_vec, p_trans)
    return trns_vec
def foraG_1(trns_vec, p_trans, lpCur):  # Subject waits
    trns_vec = transitio(lpCur, int(b), trns_vec, p_trans)
    return trns_vec
def foraG_2(trns_vec, p_trans, lpCur):  # Subject waits
    trns_vec = transitio(lpCur, int(b/2), trns_vec, p_trans)
    return trns_vec
def steaL(trns_vec, p_trans, lpCur):  # Subject waits
    trns_vec = transitio(lpCur, d, trns_vec, p_trans)
    return trns_vec
def steaG(trns_vec, p_trans, lpCur):  # Subject waits
    trns_vec = transitio(lpCur, b, trns_vec, p_trans)
    return trns_vec
# Transition matrix
def transitio(lp, val, trns_vec, p):  # Transition vector
    if lp + val == 1 or lp + val <= 0:
        trns_vec[0] = p + trns_vec[0]
    elif lp + val >= nSta-1:
        trns_vec[-1] = p + trns_vec[-1]
    else:
        trns_vec[lp+val] = p + trns_vec[lp+val]
    return trns_vec
## Define environment
nSta = 7    # LP states
nDay = 10    # Nr. of time-points
nDec = 4    # Nr. of decisions
lpAr = np.arange(0, nSta)    # LP array
obser = int((nDay+1) * nSta) # Nr. of observations
trns = np.zeros((len(lpAr))) # Empty transition vector
## state space init
def Q_space(obs_space, nDec, a):     # State-action space
    Q = np.zeros((obs_space, nDec))
    Q[:: int(nSta), :] = a
    return Q

## Social value settings
## ws ∈ [0.5,1], wo = 1-ws
## wd: subtrahend to define
## opponent's ws by self ws:
## wd ∈ [0.5,ws]
ws = 1     # Weight Qs
wo = 1-ws    # Weight Qo
wd = ws-1  # Defines ws other

# Outcome lists
Q_lstPARE = []
Q_lstNASH = []
piLstNASH = []
eqLstNASH = []
gamez = []
# Loop through items
for itr in range(0, 1):
    # State-action space
    Qs = {}
    for i in range(int(nEnv)):
        for j in range(0, nSta):
            Qs[i,j] = Q_space(obser, nDec, 0)
    Qo = copy.deepcopy(Qs)
    # Nash sums
    Ns = {}
    for i in range(int(nEnv)):
        for j in range(0, nSta):
            Ns[i,j] = Q_space(obser, nDec, 0)
    No = copy.deepcopy(Ns)
    # Reward space
    Rs = Q_space(obser, 1*nSta, 0)
    # Policy space
    pi = {}
    for i in range(int(nEnv)):
        pi[i] = copy.deepcopy(Rs)
    # Nash policiy for self
    piN = copy.deepcopy(pi)
    # Nash equilibrium
    eqN = copy.deepcopy(pi)
    # 2x2 game evaluation
    giN = {}
    for i in range(int(nEnv)):
        giN[i] = np.empty([obser, nSta], dtype=object)
    # 2x2 games reduced for plotting
    giN2 = copy.deepcopy(giN)
    # Define rewards in reward space
    for j in range(0, int(nDay)):
        Rs[j*nSta, :] = -1
        Rs[j*nSta+1, :] = -1
        Rs[j*nSta+2, :] = 1
        Rs[j*nSta+3, :] = 1
        Rs[j*nSta+4, :] = 1
        Rs[j*nSta+5, :] = 1
        Rs[j*nSta+6, :] = 1
        # Rs[nSta*(nDay+1)-nSta] = 0
    
    ## Backwards induction
    for i_day in range(0, nDay):
        for i_sta in range(0, nSta):
            for j_sta in range(0, nSta):
                for i_env in range(0, nEnv):
                    # Probability & magnitude of gain
                    pG = slct[i_env]["1 pSuccess"][0]
                    pE = slct[i_env]["1.1 pExtra coop"][0]
                    b = slct[i_env]["2 gain magnitude"][0]
                    # Current state
                    stateS = nSta + i_day * nSta + i_sta # self
                    stateO = nSta + i_day * nSta + j_sta # other
                    # Reward vector
                    rew = Rs[i_day*nSta : i_day*nSta+nSta, j_sta]
                    ## Transition vectors for game-theoretic outcomes
                    trnsCC = copy.deepcopy(trns)
                    trnsCD = copy.deepcopy(trns)
                    trnsDC = copy.deepcopy(trns)
                    trnsDD = copy.deepcopy(trns)
                    trnoCC = copy.deepcopy(trns)
                    trnoCD = copy.deepcopy(trns)
                    trnoDC = copy.deepcopy(trns)
                    trnoDD = copy.deepcopy(trns)
                    if i_sta == 0 and j_sta == 0:
                        ## Self
                        # cooperate/cooperate
                        trnsCC = foraL(trnsCC, 1, i_sta)
                        trnsCC = trnsCC/nEnv
                        # cooperate/defect
                        trnsCD = foraL(trnsCD, 1, i_sta)
                        trnsCD = trnsCD/nEnv
                        # defect/cooperate
                        trnsDC = steaL(trnsDC, 1, i_sta)
                        trnsDC = trnsDC/nEnv
                        # defect/defect
                        trnsDD = steaL(trnsDD, 1, i_sta)
                        trnsDD = trnsDD/nEnv
                        ## Other
                        # cooperate/cooperate
                        trnoCC = foraL(trnoCC, 1, j_sta)
                        trnoCC = trnoCC/nEnv
                        # cooperate/defect
                        trnoCD = foraL(trnoCD, 1, j_sta)
                        trnoCD = trnoCD/nEnv
                        # defect/cooperate
                        trnoDC = steaL(trnoDC, 1, j_sta)
                        trnoDC = trnoDC/nEnv
                        # defect/defect
                        trnoDD = steaL(trnoDD, 1, j_sta)
                        trnoDD = trnoDD/nEnv
                    elif i_sta == 0 and j_sta != 0:
                        ## Self
                        # cooperate/cooperate
                        trnsCC = foraL(trnsCC, 1, i_sta)
                        trnsCC = trnsCC/nEnv
                        # cooperate/defect
                        trnsCD = foraL(trnsCD, 1, i_sta)
                        trnsCD = trnsCD/nEnv
                        # defect/cooperate
                        trnsDC = steaL(trnsDC, 1, i_sta)
                        trnsDC = trnsDC/nEnv
                        # defect/defect
                        trnsDD = steaL(trnsDD, 1, i_sta)
                        trnsDD = trnsDD/nEnv
                        ## Other
                        # cooperate/cooperate
                        trnoCC = foraG_1(trnoCC, pG, j_sta)
                        trnoCC = foraL(trnoCC, 1-pG, j_sta)
                        trnoCC = trnoCC/nEnv
                        # cooperate/defect
                        trnoCD = foraG_1(trnoCD, pG, j_sta)
                        trnoCD = foraL(trnoCD, 1-pG, j_sta)
                        trnoCD = trnoCD/nEnv
                        # defect/cooperate
                        trnoDC = steaL(trnoDC, 1, j_sta)
                        trnoDC = trnoDC/nEnv
                        # defect/defect
                        trnoDD = steaL(trnoDD, 1, j_sta)
                        trnoDD = trnoDD/nEnv
                    elif i_sta != 0 and j_sta == 0:
                        ## Self
                        # cooperate/cooperate
                        trnsCC = foraG_1(trnsCC, pG, j_sta)
                        trnsCC = foraL(trnsCC, 1-pG, j_sta)
                        trnsCC = trnsCC/nEnv
                        # cooperate/defect
                        trnsCD = foraG_1(trnsCD, pG, j_sta)
                        trnsCD = foraL(trnsCD, 1-pG, j_sta)
                        trnsCD = trnsCD/nEnv
                        # defect/cooperate
                        trnsDC = steaL(trnsDC, 1, j_sta)
                        trnsDC = trnsDC/nEnv
                        # defect/defect
                        trnsDD = steaL(trnsDD, 1, j_sta)
                        trnsDD = trnsDD/nEnv
                        ## Other
                        # cooperate/cooperate
                        trnoCC = foraL(trnoCC, 1, j_sta)
                        trnoCC = trnoCC/nEnv
                        # cooperate/defect
                        trnoCD = foraL(trnoCD, 1, j_sta)
                        trnoCD = trnoCD/nEnv
                        # defect/cooperate
                        trnoDC = steaL(trnoDC, 1, j_sta)
                        trnoDC = trnoDC/nEnv
                        # defect/defect
                        trnoDD = steaL(trnoDD, 1, j_sta)
                        trnoDD = trnoDD/nEnv
                    else:
                        ## Self
                        # cooperate/cooperate
                        trnsCC = foraG_2(trnsCC, pG+pE, i_sta)
                        trnsCC = foraL(trnsCC, 1-(pG+pE), i_sta)
                        trnsCC = trnsCC/nEnv
                        # cooperate/defect
                        trnsCD = foraL(trnsCD, 1, i_sta)
                        trnsCD = trnsCD/nEnv
                        # defect/cooperate
                        trnsDC = steaG(trnsDC, pG, i_sta)
                        trnsDC = steaL(trnsDC, 1-pG, i_sta)
                        trnsDC = trnsDC/nEnv
                        # defect/defect
                        trnsDD = steaL(trnsDD, 1, i_sta)
                        trnsDD = trnsDD/nEnv
                        ## Other
                        # cooperate/cooperate
                        trnoCC = foraG_2(trnoCC, pG+pE, j_sta)
                        trnoCC = foraL(trnoCC, 1-(pG+pE), j_sta)
                        trnoCC = trnoCC/nEnv
                        # cooperate/defect
                        trnoCD = foraL(trnoCD, 1, j_sta)
                        trnoCD = trnoCD/nEnv
                        # defect/cooperate
                        trnoDC = steaG(trnoDC, pG, j_sta)
                        trnoDC = steaL(trnoDC, 1-pG, j_sta)
                        trnoDC = trnoDC/nEnv
                        # defect/defect
                        trnoDD = steaL(trnoDD, 1, j_sta)
                        trnoDD = trnoDD/nEnv
                    ## Update Q space for pareto im-/balance
                    # Self
                    Qs[i_env, j_sta][stateS][0] = (ws*np.dot(trnsCC, rew)+wo*np.dot(trnoCC, rew))
                    Qs[i_env, j_sta][stateS][1] = (ws*np.dot(trnsCD, rew)+wo*np.dot(trnoDC, rew))
                    Qs[i_env, j_sta][stateS][2] = (ws*np.dot(trnsDC, rew)+wo*np.dot(trnoCD, rew))
                    Qs[i_env, j_sta][stateS][3] = (ws*np.dot(trnsDD, rew)+wo*np.dot(trnoDD, rew))
                    # Update reward matrix for self
                    Rs[stateS, j_sta] = np.max(Qs[i_env, j_sta][stateS]) + Rs[stateS, j_sta]
                    # Other
                    Qo[i_env, i_sta][stateO][0] = ((ws-wd)*np.dot(trnoCC, rew)+(wo+wd)*np.dot(trnsCC, rew))
                    Qo[i_env, i_sta][stateO][1] = ((ws-wd)*np.dot(trnoCD, rew)+(wo+wd)*np.dot(trnsDC, rew))
                    Qo[i_env, i_sta][stateO][2] = ((ws-wd)*np.dot(trnoDC, rew)+(wo+wd)*np.dot(trnsCD, rew))
                    Qo[i_env, i_sta][stateO][3] = ((ws-wd)*np.dot(trnoDD, rew)+(wo+wd)*np.dot(trnsDD, rew))
                    ## Explore strategies by comparing
                    ## T with P and R with S
                    # Self
                    if Qs[i_env, j_sta][stateS][0] != Qs[i_env, j_sta][stateS][2]:
                        Ns[i_env, j_sta][stateS][np.argmax([Qs[i_env, j_sta][stateS][0],Qs[i_env, j_sta][stateS][2]])+np.argmax([Qs[i_env, j_sta][stateS][0],Qs[i_env, j_sta][stateS][2]])] += 1
                    else:
                        Ns[i_env, j_sta][stateS][0] += 1
                        Ns[i_env, j_sta][stateS][2] += 1
                    # Other
                    if Qo[i_env, i_sta][stateO][0] != Qo[i_env, i_sta][stateO][2]:
                        No[i_env, i_sta][stateO][np.argmax([Qo[i_env, i_sta][stateO][0],Qo[i_env, i_sta][stateO][2]])+np.argmax([Qo[i_env, i_sta][stateO][0],Qo[i_env, i_sta][stateO][2]])] += 1
                    else:
                        No[i_env, i_sta][stateO][0] += 1
                        No[i_env, i_sta][stateO][2] += 1
                    ## Compare S with P
                    # Self
                    if Qs[i_env, j_sta][stateS][1] != Qs[i_env, j_sta][stateS][3]:
                        Ns[i_env, j_sta][stateS][np.argmax([Qs[i_env, j_sta][stateS][1],Qs[i_env, j_sta][stateS][3]])+1+np.argmax([Qs[i_env, j_sta][stateS][1],Qs[i_env, j_sta][stateS][3]])] += 1
                    else:
                        Ns[i_env, j_sta][stateS][1] += 1
                        Ns[i_env, j_sta][stateS][3] += 1
                    # Other
                    if Qo[i_env, i_sta][stateO][1] != Qo[i_env, i_sta][stateO][3]:
                        No[i_env, i_sta][stateO][np.argmax([Qo[i_env, i_sta][stateO][1],Qo[i_env, i_sta][stateO][3]])+1+np.argmax([Qo[i_env, i_sta][stateO][1],Qo[i_env, i_sta][stateO][3]])] += 1
                    else:
                        No[i_env, i_sta][stateO][1] += 1
                        No[i_env, i_sta][stateO][3] += 1
                    ## Strategies
                    # Self
                    C = Ns[i_env, j_sta][stateS][0] + Ns[i_env, j_sta][stateS][1]
                    D = Ns[i_env, j_sta][stateS][2] + Ns[i_env, j_sta][stateS][3]
                    piN[i_env][stateS][j_sta] = np.argmax([C,D])
                    if C == D:
                        piN[i_env][stateS][j_sta] = 2
                    # Other
                    C = No[i_env, i_sta][stateO][0] + No[i_env, i_sta][stateO][1]
                    D = No[i_env, i_sta][stateO][2] + No[i_env, i_sta][stateO][3]
                    piO = np.argmax([C,D])
                    if C == D:
                        piO = 2
                    ## Explore Nash equilibrias
                    if piN[i_env][stateS][j_sta] != 2 and piO != 2:
                        eqN[i_env][stateS][j_sta] = 1   # Strict equilibrium
                    elif piN[i_env][stateS][j_sta] != 2 and piO == 2 or piN[i_env][stateS][j_sta] == 2 and piO != 2:
                        eqN[i_env][stateS][j_sta] = 2   # Weak equilibrium
                    else:
                        eqN[i_env][stateS][j_sta] = 0   # No equilibrium
                    """
# ===========================================================================
#                   Explanation unilateral policy/strategy pi
# ===========================================================================
                    piS = argmax([Ns(R+S), Ns(P+T)])
                    0 = cooperate               # strict pi
                    
                    1 = defect                  # strict pi
                    
                    if Ns(R+S) == Ns(P+T):
                        piS = 2 = indifferent   # no strict pi
# ===========================================================================
#                   Explanation Nash equilibrium
# ===========================================================================
                    if piS == strict pi and piO == strict pi:
                        strict Nash equilibrium
                        
                    if piS == strict pi and piO != strict pi:
                        weak Nash equilibrium
                        
                    if piS != strict pi and piO != strict pi:
                        no equilibrium
                    """
                    ## State game types
                    ## by rank of RTPS
                    if round(Qs[i_env, j_sta][stateS][2], 3) > round(Qs[i_env, j_sta][stateS][0], 3) > round(Qs[i_env, j_sta][stateS][3], 3) > round(Qs[i_env, j_sta][stateS][1], 3):
                        games = "prisoner_dilemma"
                    elif round(Qs[i_env, j_sta][stateS][2], 3) > round(Qs[i_env, j_sta][stateS][3], 3) > round(Qs[i_env, j_sta][stateS][0], 3) > round(Qs[i_env, j_sta][stateS][1], 3):
                        games = "deadlock"
                    elif round(Qs[i_env, j_sta][stateS][2], 3) > round(Qs[i_env, j_sta][stateS][3], 3) > round(Qs[i_env, j_sta][stateS][1], 3) > round(Qs[i_env, j_sta][stateS][0], 3):
                        games = "compromise"
                    elif round(Qs[i_env, j_sta][stateS][2], 3) > round(Qs[i_env, j_sta][stateS][1], 3) > round(Qs[i_env, j_sta][stateS][3], 3) > round(Qs[i_env, j_sta][stateS][0], 3):
                        games = "hero"
                    elif round(Qs[i_env, j_sta][stateS][2], 3) > round(Qs[i_env, j_sta][stateS][1], 3) > round(Qs[i_env, j_sta][stateS][0], 3) > round(Qs[i_env, j_sta][stateS][3], 3):
                        games = "battle"
                    elif round(Qs[i_env, j_sta][stateS][2], 3) > round(Qs[i_env, j_sta][stateS][0], 3) > round(Qs[i_env, j_sta][stateS][1], 3) > round(Qs[i_env, j_sta][stateS][3], 3):
                        games = "chicken"
                    elif round(Qs[i_env, j_sta][stateS][0], 3) > round(Qs[i_env, j_sta][stateS][2], 3) > round(Qs[i_env, j_sta][stateS][3], 3) > round(Qs[i_env, j_sta][stateS][1], 3):
                        games = "stag_hunt"
                    elif round(Qs[i_env, j_sta][stateS][0], 3) > round(Qs[i_env, j_sta][stateS][3], 3) > round(Qs[i_env, j_sta][stateS][2], 3) > round(Qs[i_env, j_sta][stateS][1], 3):
                        games = "assurance"
                    elif round(Qs[i_env, j_sta][stateS][0], 3) > round(Qs[i_env, j_sta][stateS][3], 3) > round(Qs[i_env, j_sta][stateS][1], 3) > round(Qs[i_env, j_sta][stateS][2], 3):
                        games = "coordination"
                    elif round(Qs[i_env, j_sta][stateS][0], 3) > round(Qs[i_env, j_sta][stateS][1], 3) > round(Qs[i_env, j_sta][stateS][3], 3) > round(Qs[i_env, j_sta][stateS][2], 3):
                        games = "peace"
                    elif round(Qs[i_env, j_sta][stateS][0], 3) > round(Qs[i_env, j_sta][stateS][1], 3) > round(Qs[i_env, j_sta][stateS][2], 3) > round(Qs[i_env, j_sta][stateS][3], 3):
                        games = "harmony"
                    elif round(Qs[i_env, j_sta][stateS][0], 3) > round(Qs[i_env, j_sta][stateS][2], 3) > round(Qs[i_env, j_sta][stateS][1], 3) > round(Qs[i_env, j_sta][stateS][3], 3):
                        games = "concord"
                    # inverted orders
                    elif round(Qs[i_env, j_sta][stateS][2], 3) < round(Qs[i_env, j_sta][stateS][0], 3) < round(Qs[i_env, j_sta][stateS][3], 3) < round(Qs[i_env, j_sta][stateS][1], 3):
                        games = "invert prisoner_dilemma"
                    elif round(Qs[i_env, j_sta][stateS][2], 3) < round(Qs[i_env, j_sta][stateS][3], 3) < round(Qs[i_env, j_sta][stateS][0], 3) < round(Qs[i_env, j_sta][stateS][1], 3):
                        games = "invert deadlock"
                    elif round(Qs[i_env, j_sta][stateS][2], 3) < round(Qs[i_env, j_sta][stateS][3], 3) < round(Qs[i_env, j_sta][stateS][1], 3) < round(Qs[i_env, j_sta][stateS][0], 3):
                        games = "invert compromise"
                    elif round(Qs[i_env, j_sta][stateS][2], 3) < round(Qs[i_env, j_sta][stateS][1], 3) < round(Qs[i_env, j_sta][stateS][3], 3) < round(Qs[i_env, j_sta][stateS][0], 3):
                        games = "invert hero"
                    elif round(Qs[i_env, j_sta][stateS][2], 3) < round(Qs[i_env, j_sta][stateS][1], 3) < round(Qs[i_env, j_sta][stateS][0], 3) < round(Qs[i_env, j_sta][stateS][3], 3):
                        games = "invert battle"
                    elif round(Qs[i_env, j_sta][stateS][2], 3) < round(Qs[i_env, j_sta][stateS][0], 3) < round(Qs[i_env, j_sta][stateS][1], 3) < round(Qs[i_env, j_sta][stateS][3], 3):
                        games = "invert chicken"
                    elif round(Qs[i_env, j_sta][stateS][0], 3) < round(Qs[i_env, j_sta][stateS][2], 3) < round(Qs[i_env, j_sta][stateS][3], 3) < round(Qs[i_env, j_sta][stateS][1], 3):
                        games = "invert stag_hunt"
                    elif round(Qs[i_env, j_sta][stateS][0], 3) < round(Qs[i_env, j_sta][stateS][3], 3) < round(Qs[i_env, j_sta][stateS][2], 3) < round(Qs[i_env, j_sta][stateS][1], 3):
                        games = "invert assurance"
                    elif round(Qs[i_env, j_sta][stateS][0], 3) < round(Qs[i_env, j_sta][stateS][3], 3) < round(Qs[i_env, j_sta][stateS][1], 3) < round(Qs[i_env, j_sta][stateS][2], 3):
                        games = "invert coordination"
                    elif round(Qs[i_env, j_sta][stateS][0], 3) < round(Qs[i_env, j_sta][stateS][1], 3) < round(Qs[i_env, j_sta][stateS][3], 3) < round(Qs[i_env, j_sta][stateS][2], 3):
                        games = "invert peace"
                    elif round(Qs[i_env, j_sta][stateS][0], 3) < round(Qs[i_env, j_sta][stateS][1], 3) < round(Qs[i_env, j_sta][stateS][2], 3) < round(Qs[i_env, j_sta][stateS][3], 3):
                        games = "invert harmony"
                    elif round(Qs[i_env, j_sta][stateS][0], 3) < round(Qs[i_env, j_sta][stateS][2], 3) < round(Qs[i_env, j_sta][stateS][1], 3) < round(Qs[i_env, j_sta][stateS][3], 3):
                        games = "invert concord"
                    else:
                        games = "SHARED" 
                    giN2[i_env][stateS][j_sta] = games
    # Append to outcome lists
    Q_lstPARE.append(Qs)
    Q_lstNASH.append(Ns)
    piLstNASH.append(piN)
    eqLstNASH.append(eqN)
    gamez.append(giN2)
"""
# ===========================================================================
# Explanation of outcome tables
# ===========================================================================
Rows = self_state, columns = other_state. Time points are accounted for in the rows
so that  self_stets.max() = len(nLp)*len(nTp), whereas other_state.max() = len(nLp).
To calculate deviating OP for opponent, "God" parameter w has to be adjusted to the
player01's preference weighting and player02's state has to be taken account for as
'other_state'. This can be done for both players parallel and the different OPs can
be read in the resulting tables by indexing into LP and time_point for self, and LP
for the other player.
"""

# =============================================================================
# Visualizing results for one forest conditions
# =============================================================================
## Game types
# Manipulate data frame for visualization
for it in range(0, len(gamez)):
    for ie in range(0, int(nEnv)):
        gamez[it][ie] = pd.DataFrame(gamez[itr][ie], columns = ['state other: ' + str(j-1) for j in range(0, nSta)])
        gamez[it][ie].rename(columns = {'state other: -1':'state other: 0 in'}, inplace = True)
        gamez[it][ie].rename(columns = {'state other: 0':'state other: 0 out'}, inplace = True)
# Select weather type to visualize
gamz = gamez[0][0]
# Map games to numeric
mymap = {'prisoner_dilemma':1, 'invert prisoner_dilemma':1, 'deadlock':2, 'invert deadlock':2, 'compromise':3, 'invert compromise':3,
          'hero':4, 'invert hero':4, 'battle':5, 'invert battle':5, 'chicken':6, 'invert chicken':6,
          'stag_hunt':7, 'invert stag_hunt':7, 'assurance':8, 'invert assurance':8, 'coordination':9, 'invert coordination':9,
          'peace':10, 'invert peace':10, 'harmony':11, 'invert harmony':11, 'concord':12, 'invert concord':12, 'SHARED':13}
gamz2 = gamz.applymap(lambda s: mymap.get(s) if s in mymap else s)
# Factor levels
fc = [gamz2[i].unique().tolist() for i in gamz2.columns]
fc2 = fc[0]+fc[1]+fc[2]+fc[3]+fc[4]
fc3 = list(set(fc2))
import math
fc4 = [int(item)-1 for item in fc3 if not(math.isnan(item)) == True]
fc5 = [item for item in fc3 if not(math.isnan(item)) == True]
fc4.sort()
fc5.sort()
allFacs = ['prisoner_dilemma', 'deadlock', 'compromise', 'hero', 'battle', 'chicken',
                          'stag_hunt', 'assurance', 'coordination', 'peace', 'harmony', 'concord',
                          'shared_ranks']
factor = []
for el in fc4:
    factor.append(allFacs[el])
# Manipulate index
gamz2['LP_self'] = "LP" + str(0)
gamz2['LP_self'][2::nSta] = "LP" + str(1)
gamz2['LP_self'][3::nSta] = "LP" + str(2)
gamz2['LP_self'][4::nSta] = "LP" + str(3)
gamz2['LP_self'][5::nSta] = "LP" + str(4)
gamz2['LP_self'][6::nSta] = "LP" + str(5)
gamz2['time_points'] = "t-" + str(0)
gamz2['time_points'][nSta:2*nSta] = "t-" + str(1)
gamz2['time_points'][2*nSta:3*nSta] = "t-" + str(2)
gamz2['time_points'][3*nSta:4*nSta] = "t-" + str(3)
gamz2['time_points'][4*nSta:5*nSta] = "t-" + str(4)
gamz2['time_points'][5*nSta:6*nSta] = "t-" + str(5)
gamz2['time_points'][6*nSta:7*nSta] = "t-" + str(6)
gamz2['time_points'][7*nSta:8*nSta] = "t-" + str(7)
gamz2['time_points'][8*nSta:9*nSta] = "t-" + str(8)
gamz2['time_points'][9*nSta:10*nSta] = "t-" + str(9)
gamz2['time_points'][10*nSta:11*nSta] = "t-" + str(10)
gamz2['states_self'] = gamz2['LP_self'] + " " +gamz2['time_points']
gamz2 = gamz2.set_index('states_self')
gamz2.drop(columns=gamz2.columns[-1], axis=1, inplace=True)
gamz2.drop(columns=gamz2.columns[-1], axis=1, inplace=True)
# crop t0
gamz2 = gamz2.iloc[nSta:]
## Plot
# Set publication level params
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
## Canvas
fig, ax = plt.subplots(figsize=(16, 4), 
            dpi = 600)

## Color map
# Bar has to be created manually in separate 
# script to match the state outputs
cSpace = ['#1f77b4', '#aec7e8', '#ffbb78', '#98df8a', '#d62728', '#9467bd', '#8c564b', '#c49c94', '#f7b6d2', '#c7c7c7', '#bcbd22', '#17becf', '#9edae5']
cMap = []
for el in fc4:
    cMap.append(cSpace[el])

# Create axis
ax = sns.heatmap(gamz2.T, annot=False, cmap = cMap, linewidths=.5, linecolor='black', cbar=False)
ax.tick_params(axis='x',which='major',direction='out',length=10,width=2,color='black',pad=15,labelsize=15,labelcolor='black',
                labelrotation=15)
ax.tick_params(axis='y',which='major',direction='out',length=10,width=2,color='black',pad=15,labelsize=15,labelcolor='black',
                labelrotation=15)
ax.set_xlabel('states self', fontsize = 17)
# Flip y and x axis
ax.invert_yaxis()
# Set lines for time points
b, t = plt.ylim()
ax.vlines(x = nSta, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*2, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*3, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*4, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*5, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*6, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*7, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*8, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*9, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*10, ymin = b, ymax = t, colors = 'red', lw = 3)
# Customize tick frequency
every_nth = nSta
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)
# Title
plt.title("Game types with ws = " + str(ws), size = 16)

# =============================================================================
# # Nash policies
# =============================================================================
# manipulate data for visualization
for it in range(0, len(piLstNASH)):
    for ie in range(0, int(nEnv)):
        piLstNASH[it][ie] = pd.DataFrame(piLstNASH[itr][ie], columns = ['state other: ' + str(j-1) for j in range(0, nSta)])
        piLstNASH[it][ie].rename(columns = {'state other: -1':'state other: 0 in'}, inplace = True)
        piLstNASH[it][ie].rename(columns = {'state other: 0':'state other: 0 out'}, inplace = True)      
# Select weather type to visualize
gamn = piLstNASH[0][0]
mymap = {float(0):0, float(1):1, float(2):2}
# Map outcomes to choice equilibrias
gamn2 = gamn.applymap(lambda s: mymap.get(s) if s in mymap else s)
# Factor levels
fc = [gamn2[i].unique().tolist() for i in gamn2.columns]
fc2 = fc[0]+fc[1]+fc[2]+fc[3]+fc[4]
fc3 = list(set(fc2))
import math
fc5 = [item for item in fc3 if not(math.isnan(item)) == True]
fc5.sort()
allFacs = ["cooperate", "defect", "indifferent"]
factor = []
for el in fc5:
    factor.append(allFacs[el])

## Color map
# Bar has to be created manually in separate 
# script to match the state outputs
cSpace = ['#393b79', '#b5cf6b', '#de9ed6']
cMap = []
for el in fc5:
    cMap.append(cSpace[el])

# Manipulate index
gamn2['LP_self'] = "LP" + str(0)
gamn2['LP_self'][2::nSta] = "LP" + str(1)
gamn2['LP_self'][3::nSta] = "LP" + str(2)
gamn2['LP_self'][4::nSta] = "LP" + str(3)
gamn2['LP_self'][5::nSta] = "LP" + str(4)
gamn2['LP_self'][6::nSta] = "LP" + str(5)
gamn2['time_points'] = "t-" + str(0)
gamn2['time_points'][nSta:2*nSta] = "t-" + str(1)
gamn2['time_points'][2*nSta:3*nSta] = "t-" + str(2)
gamn2['time_points'][3*nSta:4*nSta] = "t-" + str(3)
gamn2['time_points'][4*nSta:5*nSta] = "t-" + str(4)
gamn2['time_points'][5*nSta:6*nSta] = "t-" + str(5)
gamn2['time_points'][6*nSta:7*nSta] = "t-" + str(6)
gamn2['time_points'][7*nSta:8*nSta] = "t-" + str(7)
gamn2['time_points'][8*nSta:9*nSta] = "t-" + str(8)
gamn2['time_points'][9*nSta:10*nSta] = "t-" + str(9)
gamn2['time_points'][10*nSta:11*nSta] = "t-" + str(10)
gamn2['states_self'] = gamn2['LP_self'] + " " +gamn2['time_points']
gamn2 = gamn2.set_index('states_self')
gamn2.drop(columns=gamn2.columns[-1], axis=1, inplace=True)
gamn2.drop(columns=gamn2.columns[-1], axis=1, inplace=True)
# Crop t0
gamn2 = gamn2.iloc[nSta:]
## Plot
# Canvas
fig, ax = plt.subplots(figsize=(16, 4), 
            dpi = 600)
# Create axis
ax = sns.heatmap(gamn2.T, annot=False, cmap = cMap, linewidths=.5, linecolor='black', cbar=False)
ax.tick_params(axis='x',which='major',direction='out',length=10,width=2,color='black',pad=15,labelsize=15,labelcolor='black',
                labelrotation=15)
ax.tick_params(axis='y',which='major',direction='out',length=10,width=2,color='black',pad=15,labelsize=15,labelcolor='black',
                labelrotation=15)
ax.set_xlabel('states self', fontsize = 17)
# Flip y and x axis
ax.invert_yaxis()
# Set lines for time points
b, t = plt.ylim()
ax.vlines(x = nSta, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*2, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*3, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*4, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*5, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*6, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*7, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*8, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*9, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*10, ymin = b, ymax = t, colors = 'red', lw = 3)
# Customize tick frequency
every_nth = nSta
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)   
# Title
plt.title("Policies according to Nash with ws = " + str(ws), size = 16)

# =============================================================================
# # Nash equilibria
# =============================================================================
# manipulate data for visualization
for it in range(0, len(eqLstNASH)):
    for ie in range(0, int(nEnv)):
        eqLstNASH[it][ie] = pd.DataFrame(eqLstNASH[itr][ie], columns = ['state other: ' + str(j-1) for j in range(0, nSta)])
        eqLstNASH[it][ie].rename(columns = {'state other: -1':'state other: 0 in'}, inplace = True)
        eqLstNASH[it][ie].rename(columns = {'state other: 0':'state other: 0 out'}, inplace = True)      
# Select weather type to visualize
equi = eqLstNASH[0][0]
mymap = {float(0):0, float(1):2, float(2):1}
# Map outcomes to choice equilibrias
equi2 = equi.applymap(lambda s: mymap.get(s) if s in mymap else s)
# Factor levels
fc = [equi2[i].unique().tolist() for i in equi2.columns]
fc2 = fc[0]+fc[1]+fc[2]+fc[3]+fc[4]
fc3 = list(set(fc2))
import math
fc5 = [item for item in fc3 if not(math.isnan(item)) == True]
fc5.sort()
allFacs = ["cooperate", "defect", "indifferent"]
factor = []
for el in fc5:
    factor.append(allFacs[el])

## Color map
# Bar has to be created manually in separate 
# script to match the state outputs
cSpace = ['#3182bd', '#fdae6b', '#d9d9d9']
cMap = []
for el in fc5:
    cMap.append(cSpace[el])

# Manipulate index
equi2['LP_self'] = "LP" + str(0)
equi2['LP_self'][2::nSta] = "LP" + str(1)
equi2['LP_self'][3::nSta] = "LP" + str(2)
equi2['LP_self'][4::nSta] = "LP" + str(3)
equi2['LP_self'][5::nSta] = "LP" + str(4)
equi2['LP_self'][6::nSta] = "LP" + str(5)
equi2['time_points'] = "t-" + str(0)
equi2['time_points'][nSta:2*nSta] = "t-" + str(1)
equi2['time_points'][2*nSta:3*nSta] = "t-" + str(2)
equi2['time_points'][3*nSta:4*nSta] = "t-" + str(3)
equi2['time_points'][4*nSta:5*nSta] = "t-" + str(4)
equi2['time_points'][5*nSta:6*nSta] = "t-" + str(5)
equi2['time_points'][6*nSta:7*nSta] = "t-" + str(6)
equi2['time_points'][7*nSta:8*nSta] = "t-" + str(7)
equi2['time_points'][8*nSta:9*nSta] = "t-" + str(8)
equi2['time_points'][9*nSta:10*nSta] = "t-" + str(9)
equi2['time_points'][10*nSta:11*nSta] = "t-" + str(10)
equi2['states_self'] = equi2['LP_self'] + " " +equi2['time_points']
equi2 = equi2.set_index('states_self')
equi2.drop(columns=equi2.columns[-1], axis=1, inplace=True)
equi2.drop(columns=equi2.columns[-1], axis=1, inplace=True)
# Crop t0
equi2 = equi2.iloc[nSta:]
## Plot
# Canvas
fig, ax = plt.subplots(figsize=(16, 4), 
            dpi = 600)
# Create axis
ax = sns.heatmap(equi2.T, annot=False, cmap = cMap, linewidths=.5, linecolor='black', cbar=False)
ax.tick_params(axis='x',which='major',direction='out',length=10,width=2,color='black',pad=15,labelsize=15,labelcolor='black',
                labelrotation=15)
ax.tick_params(axis='y',which='major',direction='out',length=10,width=2,color='black',pad=15,labelsize=15,labelcolor='black',
                labelrotation=15)
ax.set_xlabel('states self', fontsize = 17)
ax.invert_yaxis()
# Set lines for time points
b, t = plt.ylim()
ax.vlines(x = nSta, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*2, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*3, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*4, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*5, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*6, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*7, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*8, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*9, ymin = b, ymax = t, colors = 'red', lw = 3)
ax.vlines(x = nSta*10, ymin = b, ymax = t, colors = 'red', lw = 3)
# Customize tick frequency
every_nth = nSta
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)
# Title
plt.title("Nash equilibria with player 1: ws = " + str(ws) + " and player 2: ws = " + str(round(ws-wd, 2)), size = 16)