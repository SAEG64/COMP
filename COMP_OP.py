#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:09:38 2022

@author: sergej
"""
import numpy as np
# import pandas as pd
import copy

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
# Markov Decision Process
# =============================================================================
# Define environment
nSta = 7    # LP states
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
def Q_space(obs_space, nDec):     # State-action space
    Q = np.zeros((obs_space, nDec))
    Q[:: int(nDay+1), :] = -1
    return Q

Q_lst = []
piLst = []
for itr in range(0, len(slct)):
    
    # State-action space
    Qs = {}
    for i in range(int(nEnv)):
        Qs[i] = Q_space(obser, nDec)
    # Reward spaces
    Rs = Q_space(obser, 1)
    # Policy spaces
    pi = {}
    for i in range(int(nEnv)):
        pi[i] = copy.deepcopy(Rs)
        pi[i][:: int(nDay+1), :] = np.nan
        
    for i_day in range(0, nDay):
        for i_sta in range(0, nSta-1):
            for i_env in range(0, nEnv):
                
                # Probability & magnitude of gain
                pG = slct[itr]["1 pSuccess"][i_env]
                pE = slct[itr]["1.1 pExtra coop"][i_env]
                b = slct[itr]["2 gain magnitude"][i_env]
                # Current state
                lpCur = i_sta + 1
                state = nSta + i_day * nSta + lpCur
                
                # Reward vector
                rew = Rs[i_day*nSta : i_day*nSta+nSta]
                
                # cooperate/cooperate
                trnsCC = copy.deepcopy(trns)
                trnsCC = foraG(trnsCC, pG+pE, lpCur)
                trnsCC = foraL(trnsCC, 1-(pG+pE), lpCur)
                trnsCC = trnsCC/2
                # cooperate/defect
                trnsCD = copy.deepcopy(trns)
                trnsCD = foraL(trnsCD, 1, lpCur)
                trnsCD = trnsCD/2
                # defect/cooperate
                trnsDC = copy.deepcopy(trns)
                trnsDC = steaG(trnsDC, pG, lpCur)
                trnsDC = steaL(trnsDC, 1-pG, lpCur)
                trnsDC = trnsDC/2
                # defect/defect
                trnsDD = copy.deepcopy(trns)
                trnsDD = steaL(trnsDD, 1, lpCur)
                trnsDD = trnsDD/2
                
                # Update Q and R spaces
                Qs[i_env][state][0] = np.dot(trnsCC, rew)
                Qs[i_env][state][1] = np.dot(trnsCD, rew)
                Qs[i_env][state][2] = np.dot(trnsDC, rew)
                Qs[i_env][state][3] = np.dot(trnsDD, rew)
                Rs[state] = np.max(Qs[i_env][state]) + Rs[state]
                # Update policy
                pi[i_env][state] = np.argmax(Qs[i_env][state])
                
    # Fill value and policy trables
    Q_lst.append(Qs)
    piLst.append(pi)
                
# Evaluate OP weather discrimination
tot = 0
for i in range(0, len(piLst)):
    l = np.count_nonzero(piLst[i][0] == 2)
    r = np.count_nonzero(piLst[i][1] == 2)
    if abs(l-r) > 5:
        tot += 1
print("ratio of forests where weathers differ significantly: ", tot/len(slct))























































