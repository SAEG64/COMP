#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:09:38 2022

@author: sergej

2-player foraging with option "forage" and "wait/steal"
"""
import numpy as np
# import pandas as pd
import copy
from scipy.stats import rankdata
import pandas as pd

# =============================================================================
# Item selection
# =============================================================================
# Compute expected gain
def comp_EG(pG, b, c):
    pE = (1-pG)/2
    r = (pG+pE)*b+c
    s = c
    t = pG*b+d
    p = d
    return r, s, t, p, pE

# Variables
pRaw = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
bRaw = np.array([2, 3, 4])
# Constants
nEnv = 2
c = -2
d = -1

# All item configurations - single weather
items = []
for i in range(0, len(pRaw)):
    for z in range(i, len(pRaw)-1):
        for x in range(0, len(bRaw)):
            
            # weather p and magnitude combinations
            pL = pRaw[i]
            pRw2 = np.delete(pRaw, i)
            pR = pRw2[z]
            envi = [pL, pR]
            b = bRaw[x]
            
            # Append parameters and and expected payoffs
            tabl = {"1 pSuccess":[], "1.1 pExtra coop":[], "2 gain magnitude":[],
                    "3 r":[], "4 s":[], "5 t":[], "6 p":[]}
            for i_env in range(0, nEnv):
                pG = envi[i_env]
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
    if items[i]["5 t"][0] > items[i]["3 r"][0] > items[i]["6 p"][0] > items[i]["4 s"][0] and items[i]["5 t"][1] > items[i]["3 r"][1] > items[i]["6 p"][1] > items[i]["4 s"][1]:
        slct.append(items[i])

# slctR = pd.DataFrame(slct)
# slctR.to_csv('/home/sergej/Documents/academics/dnhi/projects/prisoner_forests.csv', index = False)


# =============================================================================
# Markov Decision Process - selfish
# =============================================================================
# Define environment
nSta = 5    # LP states
nDay = 6    # Nr. of time-points
nDec = 4    # Nr. of decisions
lpAr = np.arange(0, nSta)    # LP array
obser = int((nDay+1) * nSta) # Nr. of observations
trns = np.zeros((len(lpAr))) # Empty transition vector

# Choice options & transitions
def foraL(trns_vec, p_trans, lpCur):  # Subject waits
    trns_vec = transitio(lpCur, c, trns_vec, p_trans)
    return trns_vec
def foraG(trns_vec, p_trans, lpCur):  # Subject waits
    trns_vec = transitio(lpCur, b+d, trns_vec, p_trans)
    return trns_vec
# def foraS(trns_vec, p_trans, lpCur):  # Subject waits
#     trns_vec = transitio(lpCur, c+d, trns_vec, p_trans)
#     return trns_vec

def steaL(trns_vec, p_trans, lpCur):  # Subject waits
    trns_vec = transitio(lpCur, d, trns_vec, p_trans)
    return trns_vec
def steaG(trns_vec, p_trans, lpCur):  # Subject waits
    trns_vec = transitio(lpCur, b+c, trns_vec, p_trans)
    return trns_vec

# Transition matrix
def transitio(lp, val, trns_vec, p):  # Transition vector
    if lp + val <= 0:
        trns_vec[0] = p + trns_vec[0]
    elif lp + val >= nSta-1:
        trns_vec[-1] = p + trns_vec[-1]
    else:
        trns_vec[lp+val] = p + trns_vec[lp+val]
    return trns_vec

# Value matrix
def Q_space(obs_space, nDec, a):     # State-action space
    Q = np.zeros((obs_space, nDec))
    Q[:: int(nSta), :] = a
    return Q

# =============================================================================
# Markov Decision Process - pareto & Nash
# =============================================================================
# Select all possible items
slct = items

# Define "god" player
w = 0.9 # weight of own choice

Q_lstPARE = []
piLstPARE = []
Q_lstNASH = []
piLstNASH = []
gameOrder = []
for itr in range(0, len(slct)):
    
    # State-action space
    Qs = {}
    for i in range(int(nEnv)):
        for j in range(0, nSta):
            Qs[i,j] = Q_space(obser, nDec, -1)
    # Nash sums
    Ns = {}
    for i in range(int(nEnv)):
        for j in range(0, nSta):
            Ns[i,j] = Q_space(obser, nDec, 0)
    # Reward spaces
    Rs = Q_space(obser, 1*nSta, -1)
    # Policy spaces
    pi = {}
    for i in range(int(nEnv)):
        pi[i] = copy.deepcopy(Rs)
        pi[i][:: int(nSta), :] = np.nan
    # Policy according to Nash
    piN = copy.deepcopy(pi)
    # Rank sort
    giN = {}
    for i in range(int(nEnv)):
        giN[i] = np.empty([obser, nSta], dtype=object)
    
    for i_day in range(0, nDay):
        for i_sta in range(0, nSta):
            for j_sta in range(0, nSta):
                for i_env in range(0, nEnv):
                    
                    # Probability & magnitude of gain
                    pG = slct[itr]["1 pSuccess"][i_env]
                    pE = slct[itr]["1.1 pExtra coop"][i_env]
                    b = slct[itr]["2 gain magnitude"][i_env]
                    # Current state
                    stateS = nSta + i_day * nSta + i_sta
                    
                    # Reward vector
                    rew = Rs[i_day*nSta : i_day*nSta+nSta, j_sta]
                    
                    ## Transition vectors for outcomes
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
                        trnsCC = steaL(trnsCC, 1, i_sta)
                        trnsCC = trnsCC/2
                        # cooperate/defect
                        trnsCD = steaL(trnsCD, 1, i_sta)
                        trnsCD = trnsCD/2
                        # defect/cooperate
                        trnsDC = steaL(trnsDC, 1, i_sta)
                        trnsDC = trnsDC/2
                        # defect/defect
                        trnsDD = steaL(trnsDD, 1, i_sta)
                        trnsDD = trnsDD/2
                        
                        ## Other
                        # cooperate/cooperate
                        trnoCC = steaL(trnoCC, 1, j_sta)
                        trnoCC = trnoCC/2
                        # cooperate/defect
                        trnoCD = steaL(trnoCD, 1, j_sta)
                        trnoCD = trnoCD/2
                        # defect/cooperate
                        trnoDC = steaL(trnoDC, 1, j_sta)
                        trnoDC = trnoDC/2
                        # defect/defect
                        trnoDD = steaL(trnoDD, 1, j_sta)
                        trnoDD = trnoDD/2
                    elif i_sta == 0 and j_sta != 0:
                        ## Self
                        # cooperate/cooperate
                        trnsCC = steaL(trnsCC, 1, i_sta)
                        trnsCC = trnsCC/2
                        # cooperate/defect
                        trnsCD = steaL(trnsCD, 1, i_sta)
                        trnsCD = trnsCD/2
                        # defect/cooperate
                        trnsDC = steaL(trnsDC, 1, i_sta)
                        trnsDC = trnsDC/2
                        # defect/defect
                        trnsDD = steaL(trnsDD, 1, i_sta)
                        trnsDD = trnsDD/2
                        
                        ## Other
                        # cooperate/cooperate
                        trnoCC = foraG(trnoCC, pG, j_sta)
                        trnoCC = foraL(trnoCC, 1-pG, j_sta)
                        trnoCC = trnoCC/2
                        # cooperate/defect
                        trnoCD = foraG(trnoCD, pG, j_sta)
                        trnoCD = foraL(trnoCD, 1-pG, j_sta)
                        trnoCD = trnoCD/2
                        # defect/cooperate
                        trnoDC = steaL(trnoDC, 1, j_sta)
                        trnoDC = trnoDC/2
                        # defect/defect
                        trnoDD = steaL(trnoDD, 1, j_sta)
                        trnoDD = trnoDD/2
                    elif i_sta != 0 and j_sta == 0:
                        ## Self
                        # cooperate/cooperate
                        trnsCC = foraG(trnsCC, pG, i_sta)
                        trnsCC = foraL(trnsCC, 1-pG, i_sta)
                        trnsCC = trnsCC/2
                        # cooperate/defect
                        trnsCD = foraG(trnsCD, pG, i_sta)
                        trnsCD = foraL(trnsCD, 1-pG, i_sta)
                        trnsCD = trnsCD/2
                        # defect/cooperate
                        trnsDC = steaL(trnsDC, 1, i_sta)
                        trnsDC = trnsDC/2
                        # defect/defect
                        trnsDD = steaL(trnsDD, 1, i_sta)
                        trnsDD = trnsDD/2
                        
                        ## Other
                        # cooperate/cooperate
                        trnoCC = steaL(trnoCC, 1, j_sta)
                        trnoCC = trnoCC/2
                        # cooperate/defect
                        trnoCD = steaL(trnoCD, 1, j_sta)
                        trnoCD = trnoCD/2
                        # defect/cooperate
                        trnoDC = steaL(trnoDC, 1, j_sta)
                        trnoDC = trnoDC/2
                        # defect/defect
                        trnoDD = steaL(trnoDD, 1, j_sta)
                        trnoDD = trnoDD/2
                    else:
                        ## Self
                        # cooperate/cooperate
                        trnsCC = foraG(trnsCC, pG+pE, i_sta)
                        trnsCC = foraL(trnsCC, 1-(pG+pE), i_sta)
                        trnsCC = trnsCC/2
                        # cooperate/defect
                        trnsCD = foraL(trnsCD, 1, i_sta)
                        trnsCD = trnsCD/2
                        # defect/cooperate
                        trnsDC = steaG(trnsDC, pG, i_sta)
                        trnsDC = steaL(trnsDC, 1-pG, i_sta)
                        trnsDC = trnsDC/2
                        # defect/defect
                        trnsDD = steaL(trnsDD, 1, i_sta)
                        trnsDD = trnsDD/2
                        
                        ## Other
                        # cooperate/cooperate
                        trnoCC = foraG(trnoCC, pG+pE, j_sta)
                        trnoCC = foraL(trnoCC, 1-(pG+pE), j_sta)
                        trnoCC = trnoCC/2
                        # cooperate/defect
                        trnoCD = foraL(trnoCD, 1, j_sta)
                        trnoCD = trnoCD/2
                        # defect/cooperate
                        trnoDC = steaG(trnoDC, pG, j_sta)
                        trnoDC = steaL(trnoDC, 1-pG, j_sta)
                        trnoDC = trnoDC/2
                        # defect/defect
                        trnoDD = steaL(trnoDD, 1, j_sta)
                        trnoDD = trnoDD/2
                    
                    # Update Q space
                    Qs[i_env, j_sta][stateS][0] = w*np.dot(trnsCC, rew)+(1-w)*np.dot(trnoCC, rew)
                    Qs[i_env, j_sta][stateS][1] = w*np.dot(trnsCD, rew)+(1-w)*np.dot(trnoDC, rew)
                    Qs[i_env, j_sta][stateS][2] = w*np.dot(trnsDC, rew)+(1-w)*np.dot(trnoCD, rew)
                    Qs[i_env, j_sta][stateS][3] = w*np.dot(trnsDD, rew)+(1-w)*np.dot(trnoDD, rew)
                    Qs[i_env, j_sta][:: int(nSta), :] = -1 # Reset absorbing death states
                    
                    # Explore Nash equilibrias
                    if Qs[i_env, j_sta][stateS][0] != Qs[i_env, j_sta][stateS][1]:
                        Ns[i_env, j_sta][stateS][np.argmax([Qs[i_env, j_sta][stateS][0],Qs[i_env, j_sta][stateS][1]])] += 1
                    else:
                        Ns[i_env, j_sta][stateS][0] += 0.5
                        Ns[i_env, j_sta][stateS][1] += 0.5
                    if Qs[i_env, j_sta][stateS][0] != Qs[i_env, j_sta][stateS][2]:
                        Ns[i_env, j_sta][stateS][np.argmax([Qs[i_env, j_sta][stateS][0],Qs[i_env, j_sta][stateS][2]])+np.argmax([Qs[i_env, j_sta][stateS][0],Qs[i_env, j_sta][stateS][2]])] += 1
                    else:
                        Ns[i_env, j_sta][stateS][0] += 0.5
                        Ns[i_env, j_sta][stateS][2] += 0.5
                    if Qs[i_env, j_sta][stateS][1] != Qs[i_env, j_sta][stateS][3]:
                        Ns[i_env, j_sta][stateS][np.argmax([Qs[i_env, j_sta][stateS][1],Qs[i_env, j_sta][stateS][3]])+1+np.argmax([Qs[i_env, j_sta][stateS][1],Qs[i_env, j_sta][stateS][3]])] += 1
                    else:
                        Ns[i_env, j_sta][stateS][1] += 0.5
                        Ns[i_env, j_sta][stateS][3] += 0.5
                    if Qs[i_env, j_sta][stateS][2] != Qs[i_env, j_sta][stateS][3]:
                        Ns[i_env, j_sta][stateS][np.argmax([Qs[i_env, j_sta][stateS][2],Qs[i_env, j_sta][stateS][3]])+2] += 1
                    else:
                        Ns[i_env, j_sta][stateS][2] += 0.5
                        Ns[i_env, j_sta][stateS][3] += 0.5
                    Ns[i_env, j_sta][:: int(nSta), :] = np.nan # Reset absorbing death states of self
                    
                    # Update reward matrix
                    Rs[:: int(nSta), :] = -1
                    Rs[stateS, j_sta] = np.max(Qs[i_env, j_sta][stateS]) + Rs[stateS, j_sta]
                    
                    # Update policy pareto
                    pi[i_env][stateS][j_sta] = np.argmax(Qs[i_env, j_sta][stateS])
                    if Qs[i_env, j_sta][stateS][0] == Qs[i_env, j_sta][stateS][1] == Qs[i_env, j_sta][stateS][2] == Qs[i_env, j_sta][stateS][3] == -1:
                        pi[i_env][stateS][j_sta] = np.nan
                    elif Qs[i_env, j_sta][stateS][0] == Qs[i_env, j_sta][stateS][1] == Qs[i_env, j_sta][stateS][2] == Qs[i_env, j_sta][stateS][3]:
                        pi[i_env][stateS][j_sta] = 3210
                    elif Qs[i_env, j_sta][stateS][0] == Qs[i_env, j_sta][stateS][1] == Qs[i_env, j_sta][stateS][2] and np.argmax(Qs[i_env, j_sta][stateS]) == 0:
                        pi[i_env][stateS][j_sta] = 210
                    elif Qs[i_env, j_sta][stateS][1] == Qs[i_env, j_sta][stateS][2] == Qs[i_env, j_sta][stateS][3] and np.argmax(Qs[i_env, j_sta][stateS]) == 1:
                        pi[i_env][stateS][j_sta] = 321
                    elif Qs[i_env, j_sta][stateS][2] == Qs[i_env, j_sta][stateS][3] == Qs[i_env, j_sta][stateS][0] and np.argmax(Qs[i_env, j_sta][stateS]) == 0:
                        pi[i_env][stateS][j_sta] = 320
                    elif Qs[i_env, j_sta][stateS][3] == Qs[i_env, j_sta][stateS][0] == Qs[i_env, j_sta][stateS][1]and np.argmax(Qs[i_env, j_sta][stateS]) == 0:
                        pi[i_env][stateS][j_sta] = 310
                    elif Qs[i_env, j_sta][stateS][0] == Qs[i_env, j_sta][stateS][1] and np.argmax(Qs[i_env, j_sta][stateS]) == 0:
                        pi[i_env][stateS][j_sta] = 10
                    elif Qs[i_env, j_sta][stateS][2] == Qs[i_env, j_sta][stateS][3] and np.argmax(Qs[i_env, j_sta][stateS]) == 2:
                        pi[i_env][stateS][j_sta] = 32
                    elif Qs[i_env, j_sta][stateS][0] == Qs[i_env, j_sta][stateS][2] and np.argmax(Qs[i_env, j_sta][stateS]) == 0:
                        pi[i_env][stateS][j_sta] = 20
                    elif Qs[i_env, j_sta][stateS][0] == Qs[i_env, j_sta][stateS][3] and np.argmax(Qs[i_env, j_sta][stateS]) == 0:
                        pi[i_env][stateS][j_sta] = 30
                    elif Qs[i_env, j_sta][stateS][1] == Qs[i_env, j_sta][stateS][2] and np.argmax(Qs[i_env, j_sta][stateS]) == 1: 
                        pi[i_env][stateS][j_sta] = 21
                    elif Qs[i_env, j_sta][stateS][1] == Qs[i_env, j_sta][stateS][3] and np.argmax(Qs[i_env, j_sta][stateS]) == 1:
                        pi[i_env][stateS][j_sta] = 31
                    
                    # Update policy according to Nash
                    piN[i_env][stateS][j_sta] = np.argmax(Ns[i_env, j_sta][stateS])
                    if Ns[i_env, j_sta][stateS][0] == Ns[i_env, j_sta][stateS][1] == Ns[i_env, j_sta][stateS][2] == Ns[i_env, j_sta][stateS][3] != np.nan:
                        piN[i_env][stateS][j_sta] = 3210
                    elif Ns[i_env, j_sta][stateS][0] == Ns[i_env, j_sta][stateS][1] == Ns[i_env, j_sta][stateS][2] and np.argmax(Ns[i_env, j_sta][stateS]) == 0:
                        piN[i_env][stateS][j_sta] = 210
                    elif Ns[i_env, j_sta][stateS][1] == Ns[i_env, j_sta][stateS][2] == Ns[i_env, j_sta][stateS][3] and np.argmax(Ns[i_env, j_sta][stateS]) == 1:
                        piN[i_env][stateS][j_sta] = 321
                    elif Ns[i_env, j_sta][stateS][2] == Ns[i_env, j_sta][stateS][3] == Ns[i_env, j_sta][stateS][0] and np.argmax(Ns[i_env, j_sta][stateS]) == 0:
                        piN[i_env][stateS][j_sta] = 320
                    elif Ns[i_env, j_sta][stateS][3] == Ns[i_env, j_sta][stateS][0] == Ns[i_env, j_sta][stateS][1]and np.argmax(Ns[i_env, j_sta][stateS]) == 0:
                        piN[i_env][stateS][j_sta] = 310
                    elif Ns[i_env, j_sta][stateS][0] == Ns[i_env, j_sta][stateS][1] and np.argmax(Ns[i_env, j_sta][stateS]) == 0:
                        piN[i_env][stateS][j_sta] = 10
                    elif Ns[i_env, j_sta][stateS][2] == Ns[i_env, j_sta][stateS][3] and np.argmax(Ns[i_env, j_sta][stateS]) == 2:
                        piN[i_env][stateS][j_sta] = 32
                    elif Ns[i_env, j_sta][stateS][0] == Ns[i_env, j_sta][stateS][2] and np.argmax(Ns[i_env, j_sta][stateS]) == 0:
                        piN[i_env][stateS][j_sta] = 20
                    elif Ns[i_env, j_sta][stateS][0] == Ns[i_env, j_sta][stateS][3] and np.argmax(Ns[i_env, j_sta][stateS]) == 0:
                        piN[i_env][stateS][j_sta] = 30
                    elif Ns[i_env, j_sta][stateS][1] == Ns[i_env, j_sta][stateS][2] and np.argmax(Ns[i_env, j_sta][stateS]) == 1: 
                        piN[i_env][stateS][j_sta] = 21
                    elif Ns[i_env, j_sta][stateS][1] == Ns[i_env, j_sta][stateS][3] and np.argmax(Ns[i_env, j_sta][stateS]) == 1:
                        piN[i_env][stateS][j_sta] = 31
                    piN[i_env][:: int(nSta), :] = np.nan
                    
                    # Sort and rank values - 1 = R, 2 = S, 3 = T, 4 = P
                    ordr = np.flip(np.argsort(Qs[i_env, j_sta][stateS]))
                    rank = rankdata(Qs[i_env, j_sta][stateS], method='min')
                    rank.sort()
                    
                    # Append sorted and ranked values
                    giN[i_env][stateS][j_sta] = np.append(giN[i_env][stateS][j_sta], ordr)
                    giN[i_env][stateS][j_sta] = giN[i_env][stateS][j_sta][giN[i_env][stateS][j_sta] != None]
                    giN[i_env][stateS][j_sta] = np.append(giN[i_env][stateS][j_sta], rank)
                    giN[i_env][stateS][j_sta] = np.array_split(giN[i_env][stateS][j_sta], 2)
                    
                    # Ranks sorted unique or shared
                    if giN[i_env][stateS][j_sta][1][0] == giN[i_env][stateS][j_sta][1][1] or giN[i_env][stateS][j_sta][1][0] == giN[i_env][stateS][j_sta][1][2] or giN[i_env][stateS][j_sta][1][0] == giN[i_env][stateS][j_sta][1][3] or giN[i_env][stateS][j_sta][1][1] == giN[i_env][stateS][j_sta][1][2] or giN[i_env][stateS][j_sta][1][1] == giN[i_env][stateS][j_sta][1][3] or giN[i_env][stateS][j_sta][1][2] == giN[i_env][stateS][j_sta][1][3]:
                        pool  = np.array(("shared_ranks"))
                    else:
                        pool  = np.array(("unique_sorts"))
                    
                    # Game types
                    if abs(Qs[i_env, j_sta][stateS][2]) < abs(Qs[i_env, j_sta][stateS][0]) < abs(Qs[i_env, j_sta][stateS][3]) < abs(Qs[i_env, j_sta][stateS][1]):
                        game = np.array(("prisoner_dilemma"))
                    elif abs(Qs[i_env, j_sta][stateS][2]) < abs(Qs[i_env, j_sta][stateS][3]) < abs(Qs[i_env, j_sta][stateS][0]) < abs(Qs[i_env, j_sta][stateS][1]):
                        game = np.array(("deadlock"))
                    elif abs(Qs[i_env, j_sta][stateS][2]) < abs(Qs[i_env, j_sta][stateS][3]) < abs(Qs[i_env, j_sta][stateS][1]) < abs(Qs[i_env, j_sta][stateS][0]):
                        game = np.array(("compromise"))
                    elif abs(Qs[i_env, j_sta][stateS][2]) < abs(Qs[i_env, j_sta][stateS][1]) < abs(Qs[i_env, j_sta][stateS][3]) < abs(Qs[i_env, j_sta][stateS][0]):
                        game = np.array(("hero"))
                    elif abs(Qs[i_env, j_sta][stateS][2]) < abs(Qs[i_env, j_sta][stateS][1]) < abs(Qs[i_env, j_sta][stateS][0]) < abs(Qs[i_env, j_sta][stateS][1]):
                        game = np.array(("battle"))
                    elif abs(Qs[i_env, j_sta][stateS][2]) < abs(Qs[i_env, j_sta][stateS][0]) < abs(Qs[i_env, j_sta][stateS][1]) < abs(Qs[i_env, j_sta][stateS][3]):
                        game = np.array(("chicken"))
                    elif abs(Qs[i_env, j_sta][stateS][0]) < abs(Qs[i_env, j_sta][stateS][2]) < abs(Qs[i_env, j_sta][stateS][3]) < abs(Qs[i_env, j_sta][stateS][1]):
                        game = np.array(("stag_hunt"))
                    elif abs(Qs[i_env, j_sta][stateS][0]) < abs(Qs[i_env, j_sta][stateS][3]) < abs(Qs[i_env, j_sta][stateS][2]) < abs(Qs[i_env, j_sta][stateS][1]):
                        game = np.array(("assurance"))
                    elif abs(Qs[i_env, j_sta][stateS][0]) < abs(Qs[i_env, j_sta][stateS][3]) < abs(Qs[i_env, j_sta][stateS][1]) < abs(Qs[i_env, j_sta][stateS][2]):
                        game = np.array(("coordination"))
                    elif abs(Qs[i_env, j_sta][stateS][0]) < abs(Qs[i_env, j_sta][stateS][1]) < abs(Qs[i_env, j_sta][stateS][3]) < abs(Qs[i_env, j_sta][stateS][2]):
                        game = np.array(("peace"))
                    elif abs(Qs[i_env, j_sta][stateS][0]) < abs(Qs[i_env, j_sta][stateS][1]) < abs(Qs[i_env, j_sta][stateS][2]) < abs(Qs[i_env, j_sta][stateS][3]):
                        game = np.array(("harmony"))
                    elif abs(Qs[i_env, j_sta][stateS][0]) < abs(Qs[i_env, j_sta][stateS][2]) < abs(Qs[i_env, j_sta][stateS][1]) < abs(Qs[i_env, j_sta][stateS][3]):
                        game = np.array(("concord"))
                    
                    elif abs(Qs[i_env, j_sta][stateS][2]) > abs(Qs[i_env, j_sta][stateS][0]) > abs(Qs[i_env, j_sta][stateS][3]) > abs(Qs[i_env, j_sta][stateS][1]):
                        game = np.array(("INVERT_prisoner_dilemma"))
                    elif abs(Qs[i_env, j_sta][stateS][2]) > abs(Qs[i_env, j_sta][stateS][3]) > abs(Qs[i_env, j_sta][stateS][0]) > abs(Qs[i_env, j_sta][stateS][1]):
                        game = np.array(("INVERT_deadlock"))
                    elif abs(Qs[i_env, j_sta][stateS][2]) > abs(Qs[i_env, j_sta][stateS][3]) > abs(Qs[i_env, j_sta][stateS][1]) > abs(Qs[i_env, j_sta][stateS][0]):
                        game = np.array(("INVERT_compromise"))
                    elif abs(Qs[i_env, j_sta][stateS][2]) > abs(Qs[i_env, j_sta][stateS][1]) > abs(Qs[i_env, j_sta][stateS][3]) > abs(Qs[i_env, j_sta][stateS][0]):
                        game = np.array(("INVERT_hero"))
                    elif abs(Qs[i_env, j_sta][stateS][2]) > abs(Qs[i_env, j_sta][stateS][1]) > abs(Qs[i_env, j_sta][stateS][0]) > abs(Qs[i_env, j_sta][stateS][1]):
                        game = np.array(("INVERT_battle"))
                    elif abs(Qs[i_env, j_sta][stateS][2]) > abs(Qs[i_env, j_sta][stateS][0]) > abs(Qs[i_env, j_sta][stateS][1]) > abs(Qs[i_env, j_sta][stateS][3]):
                        game = np.array(("INVERT_chicken"))
                    elif abs(Qs[i_env, j_sta][stateS][0]) > abs(Qs[i_env, j_sta][stateS][2]) > abs(Qs[i_env, j_sta][stateS][3]) > abs(Qs[i_env, j_sta][stateS][1]):
                        game = np.array(("INVERT_stag_hunt"))
                    elif abs(Qs[i_env, j_sta][stateS][0]) > abs(Qs[i_env, j_sta][stateS][3]) > abs(Qs[i_env, j_sta][stateS][2]) > abs(Qs[i_env, j_sta][stateS][1]):
                        game = np.array(("INVERT_assurance"))
                    elif abs(Qs[i_env, j_sta][stateS][0]) > abs(Qs[i_env, j_sta][stateS][3]) > abs(Qs[i_env, j_sta][stateS][1]) > abs(Qs[i_env, j_sta][stateS][2]):
                        game = np.array(("INVERT_coordination"))
                    elif abs(Qs[i_env, j_sta][stateS][0]) > abs(Qs[i_env, j_sta][stateS][1]) > abs(Qs[i_env, j_sta][stateS][3]) > abs(Qs[i_env, j_sta][stateS][2]):
                        game = np.array(("INVERT_peace"))
                    elif abs(Qs[i_env, j_sta][stateS][0]) > abs(Qs[i_env, j_sta][stateS][1]) > abs(Qs[i_env, j_sta][stateS][2]) > abs(Qs[i_env, j_sta][stateS][3]):
                        game = np.array(("INVERT_harmony"))
                    elif abs(Qs[i_env, j_sta][stateS][0]) > abs(Qs[i_env, j_sta][stateS][2]) > abs(Qs[i_env, j_sta][stateS][1]) > abs(Qs[i_env, j_sta][stateS][3]):
                        game = np.array(("INVERT_concord"))
                    else:
                        game = np.array(("special")) 
                    
                    giN[i_env][stateS][j_sta] = np.column_stack([giN[i_env][stateS][j_sta], [game, pool]])
                    giN[i_env][:: int(nSta), :][j_sta] = np.nan
    
    # Fill value and policy trables
    Q_lstPARE.append(Qs)
    piLstPARE.append(pi)
    Q_lstNASH.append(Ns)
    piLstNASH.append(piN)
    gameOrder.append(giN)

# Convert gameOrder to readable pandas data frame
for it in range(0, len(gameOrder)):
    for ie in range(0, int(nEnv)):
        gameOrder[it][ie] = pd.DataFrame(gameOrder[itr][i_env], columns = ['state' + str(j) for j in range(0, nSta)])
    
# # Evaluate OP weather discrimination
# tot = 0
# q_PareREV = []
# piPareREV = []
# q_NashREV = []
# piNashREV = []
# gaNashREV = []
# for i in range(0, len(piLstPARE)):
#     l = np.count_nonzero(piLstPARE[i][0] == 3)
#     r = np.count_nonzero(piLstPARE[i][1] == 3)
#     if abs(l-r) > 15:
#         q_PareREV.append(Q_lstPARE[i])
#         piPareREV.append(piLstPARE[i])
#         q_NashREV.append(Q_lstNASH[i])
#         piNashREV.append(piLstNASH[i])
#         gaNashREV.append(gameOrder[i])
#         tot += 1
# print("ratio of forests where weathers differ significantly: ", tot/len(slct))

                
