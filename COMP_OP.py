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

# slctR = pd.DataFrame(slct)
# slctR.to_csv('/home/sergej/Documents/academics/dnhi/projects/prisoner_forests.csv', index = False)


# =============================================================================
# Markov Decision Process - transitions and state space
# =============================================================================
# Choice options & transitions
def foraL(trns_vec, p_trans, lpCur):  # Subject waits
    trns_vec = transitio(lpCur, c, trns_vec, p_trans)
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
# Run MDP - pareto & Nash
# =============================================================================
# Select all possible items
# slct = items

# Define environment
nSta = 5    # LP states
nDay = 10    # Nr. of time-points
nDec = 4    # Nr. of decisions
lpAr = np.arange(0, nSta)    # LP array
obser = int((nDay+1) * nSta) # Nr. of observations
trns = np.zeros((len(lpAr))) # Empty transition vector
w = 0.5 # God player weight

Q_lstPARE = []
piLstPARE = []
Q_lstNASH = []
piLstNASH = []
gameOrder = []
for itr in range(0, 1):
    
    # State-action space
    Qs = {}
    for i in range(int(nEnv)):
        for j in range(0, nSta):
            Qs[i,j] = Q_space(obser, nDec, 0)
    # Nash sums
    Ns = {}
    for i in range(int(nEnv)):
        for j in range(0, nSta):
            Ns[i,j] = Q_space(obser, nDec, 0)
    # Reward space
    Rs = Q_space(obser, 1*nSta, 0)
    # Policy space
    pi = {}
    for i in range(int(nEnv)):
        pi[i] = copy.deepcopy(Rs)
        # pi[i][:: int(nSta), :] = np.nan
    # Policy according to Nash
    piN = copy.deepcopy(pi)
    # Rank sort
    giN = {}
    for i in range(int(nEnv)):
        giN[i] = np.empty([obser, nSta], dtype=object)
    # Define rewards in reward space
    for j in range(0, int(nDay)):
        Rs[j*nSta, :] = -1
        Rs[j*nSta+1, :] = 1
        Rs[j*nSta+2, :] = 1
        Rs[j*nSta+3, :] = 1
        Rs[j*nSta+4, :] = 1
        Rs[j*nSta+5, :] = 1
    
    for i_day in range(0, nDay):
        for i_sta in range(0, nSta):
            for j_sta in range(0, nSta):
                for i_env in range(0, nEnv):
                    
                    # Probability & magnitude of gain
                    pG = slct[i_env]["1 pSuccess"][0]
                    pE = slct[i_env]["1.1 pExtra coop"][0]
                    b = slct[i_env]["2 gain magnitude"][0]
                    # Current state
                    stateS = nSta + i_day * nSta + i_sta
                    
                    # Reward vector
                    rew = Rs[i_day*nSta : i_day*nSta+nSta, j_sta]
                    
                    ## Transition vectors for game theoretic outcomes
                    trnsCC = copy.deepcopy(trns)
                    trnsCD = copy.deepcopy(trns)
                    trnsDC = copy.deepcopy(trns)
                    trnsDD = copy.deepcopy(trns)
                    trnoCC = copy.deepcopy(trns)
                    trnoCD = copy.deepcopy(trns)
                    trnoDC = copy.deepcopy(trns)
                    trnoDD = copy.deepcopy(trns)
                    ## Self
                    # cooperate/cooperate
                    trnsCC = foraG_2(trnsCC, pG+pE, i_sta)
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
                    trnoCC = foraG_2(trnoCC, pG+pE, j_sta)
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
                    
                    # Update reward matrix
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
                    
                    # Sort and rank values: 0 = R, 1 = S, 2 = T, 3 = P
                    ordr = np.flip(np.argsort(Qs[i_env, j_sta][stateS]))
                    rank = rankdata(Qs[i_env, j_sta][stateS], method='min')
                    rank.sort()
                    
                    # Append sorted and ranked values
                    giN[i_env][stateS][j_sta] = np.append(giN[i_env][stateS][j_sta], ordr)
                    giN[i_env][stateS][j_sta] = giN[i_env][stateS][j_sta][giN[i_env][stateS][j_sta] != None]
                    giN[i_env][stateS][j_sta] = np.append(giN[i_env][stateS][j_sta], rank)
                    giN[i_env][stateS][j_sta] = np.array_split(giN[i_env][stateS][j_sta], 2)
                    
                    # Game types
                    if Qs[i_env, j_sta][stateS][2] > Qs[i_env, j_sta][stateS][0] > Qs[i_env, j_sta][stateS][3] > Qs[i_env, j_sta][stateS][1]:
                        game = np.array(("prisoner_dilemma"))
                    elif Qs[i_env, j_sta][stateS][2] > Qs[i_env, j_sta][stateS][3] > Qs[i_env, j_sta][stateS][0] > Qs[i_env, j_sta][stateS][1]:
                        game = np.array(("deadlock"))
                    elif Qs[i_env, j_sta][stateS][2] > Qs[i_env, j_sta][stateS][3] > Qs[i_env, j_sta][stateS][1] > Qs[i_env, j_sta][stateS][0]:
                        game = np.array(("compromise"))
                    elif Qs[i_env, j_sta][stateS][2] > Qs[i_env, j_sta][stateS][1] > Qs[i_env, j_sta][stateS][3] > Qs[i_env, j_sta][stateS][0]:
                        game = np.array(("hero"))
                    elif Qs[i_env, j_sta][stateS][2] > Qs[i_env, j_sta][stateS][1] > Qs[i_env, j_sta][stateS][0] > Qs[i_env, j_sta][stateS][3]:
                        game = np.array(("battle"))
                    elif Qs[i_env, j_sta][stateS][2] > Qs[i_env, j_sta][stateS][0] > Qs[i_env, j_sta][stateS][1] > Qs[i_env, j_sta][stateS][3]:
                        game = np.array(("chicken"))
                    elif Qs[i_env, j_sta][stateS][0] > Qs[i_env, j_sta][stateS][2] > Qs[i_env, j_sta][stateS][3] > Qs[i_env, j_sta][stateS][1]:
                        game = np.array(("stag_hunt"))
                    elif Qs[i_env, j_sta][stateS][0] > Qs[i_env, j_sta][stateS][3] > Qs[i_env, j_sta][stateS][2] > Qs[i_env, j_sta][stateS][1]:
                        game = np.array(("assurance"))
                    elif Qs[i_env, j_sta][stateS][0] > Qs[i_env, j_sta][stateS][3] > Qs[i_env, j_sta][stateS][1] > Qs[i_env, j_sta][stateS][2]:
                        game = np.array(("coordination"))
                    elif Qs[i_env, j_sta][stateS][0] > Qs[i_env, j_sta][stateS][1] > Qs[i_env, j_sta][stateS][3] > Qs[i_env, j_sta][stateS][2]:
                        game = np.array(("peace"))
                    elif Qs[i_env, j_sta][stateS][0] > Qs[i_env, j_sta][stateS][1] > Qs[i_env, j_sta][stateS][2] > Qs[i_env, j_sta][stateS][3]:
                        game = np.array(("harmony"))
                    elif Qs[i_env, j_sta][stateS][0] > Qs[i_env, j_sta][stateS][2] > Qs[i_env, j_sta][stateS][1] > Qs[i_env, j_sta][stateS][3]:
                        game = np.array(("concord"))
                    # inverted orders
                    elif Qs[i_env, j_sta][stateS][2] < Qs[i_env, j_sta][stateS][0] < Qs[i_env, j_sta][stateS][3] < Qs[i_env, j_sta][stateS][1]:
                        game = np.array(("invert prisoner_dilemma"))
                    elif Qs[i_env, j_sta][stateS][2] < Qs[i_env, j_sta][stateS][3] < Qs[i_env, j_sta][stateS][0] < Qs[i_env, j_sta][stateS][1]:
                        game = np.array(("invert deadlock"))
                    elif Qs[i_env, j_sta][stateS][2] < Qs[i_env, j_sta][stateS][3] < Qs[i_env, j_sta][stateS][1] < Qs[i_env, j_sta][stateS][0]:
                        game = np.array(("invert compromise"))
                    elif Qs[i_env, j_sta][stateS][2] < Qs[i_env, j_sta][stateS][1] < Qs[i_env, j_sta][stateS][3] < Qs[i_env, j_sta][stateS][0]:
                        game = np.array(("invert hero"))
                    elif Qs[i_env, j_sta][stateS][2] < Qs[i_env, j_sta][stateS][1] < Qs[i_env, j_sta][stateS][0] < Qs[i_env, j_sta][stateS][3]:
                        game = np.array(("invert battle"))
                    elif Qs[i_env, j_sta][stateS][2] < Qs[i_env, j_sta][stateS][0] < Qs[i_env, j_sta][stateS][1] < Qs[i_env, j_sta][stateS][3]:
                        game = np.array(("invert chicken"))
                    elif Qs[i_env, j_sta][stateS][0] < Qs[i_env, j_sta][stateS][2] < Qs[i_env, j_sta][stateS][3] < Qs[i_env, j_sta][stateS][1]:
                        game = np.array(("invert stag_hunt"))
                    elif Qs[i_env, j_sta][stateS][0] < Qs[i_env, j_sta][stateS][3] < Qs[i_env, j_sta][stateS][2] < Qs[i_env, j_sta][stateS][1]:
                        game = np.array(("invert assurance"))
                    elif Qs[i_env, j_sta][stateS][0] < Qs[i_env, j_sta][stateS][3] < Qs[i_env, j_sta][stateS][1] < Qs[i_env, j_sta][stateS][2]:
                        game = np.array(("invert coordination"))
                    elif Qs[i_env, j_sta][stateS][0] < Qs[i_env, j_sta][stateS][1] < Qs[i_env, j_sta][stateS][3] < Qs[i_env, j_sta][stateS][2]:
                        game = np.array(("invert peace"))
                    elif Qs[i_env, j_sta][stateS][0] < Qs[i_env, j_sta][stateS][1] < Qs[i_env, j_sta][stateS][2] < Qs[i_env, j_sta][stateS][3]:
                        game = np.array(("invert harmony"))
                    elif Qs[i_env, j_sta][stateS][0] < Qs[i_env, j_sta][stateS][2] < Qs[i_env, j_sta][stateS][1] < Qs[i_env, j_sta][stateS][3]:
                        game = np.array(("invert concord"))
                    else:
                        game = np.array(("SHARED")) 
                    
                    giN[i_env][stateS][j_sta] = np.column_stack([giN[i_env][stateS][j_sta], [game, ""]])
                    # Replace numeric with letter code: 0 = R, 1 = S, 2 = T, 3 = P
                    giN[i_env][stateS][j_sta][0][giN[i_env][stateS][j_sta][0] == 0] = "R"
                    giN[i_env][stateS][j_sta][0][giN[i_env][stateS][j_sta][0] == 1] = "S"
                    giN[i_env][stateS][j_sta][0][giN[i_env][stateS][j_sta][0] == 2] = "T"
                    giN[i_env][stateS][j_sta][0][giN[i_env][stateS][j_sta][0] == 3] = "P"
    # giN[i_env][:: int(nSta), :][:] = np.nan # Reset absorbing death states
    
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

""" Note:   Rows = self_state, columns = other_state. Time points are accounted for in the rows
            so that  self_stets.max() = len(nLp)*len(nTp), whereas other_state.max() = len(nLp).
            To calculate deviating OP for opponent, "God" parameter w has to be adjusted to the
            player01's preference weighting and player02's state has to be taken account for as
            'other_state'. This can be done for both players parallel and the different OPs can
            be read in the resulting tables by indexing into LP and time_point for self, and LP
            for the other player. """
    
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

                
