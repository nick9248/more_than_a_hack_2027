"""Evidence-weighted transitions with optional soft short-spell costs and factorisation."""
import numpy as np
from numba import njit
from .data import blocks, X
from scipy.special import expit, logit

def emission_probabilities(p_live, p_home, seen, missing_team_temperature=1.0):
    # Flatten uncertain team evidence during an occlusion; do not alter live/dead evidence.
    temperature = np.where(seen, 1., missing_team_temperature)
    home = expit(logit(np.clip(p_home, 1e-5, 1-1e-5)) * temperature)
    live = np.clip(p_live, 1e-5, 1-1e-5)
    return np.column_stack([live * home, live * (1-home), 1-live])



@njit(cache=True)
def path3(e, team, enter_dead, leave_dead):
    n = len(e)
    back = np.zeros((n, 3), np.int8)
    scores = e[0].astype(np.float64)
    for i in range(1, n):
        update = np.empty(3, np.float64)
        for dest in range(3):
            best, origin = scores[dest], dest
            for src in range(3):
                if src == dest:
                    continue
                cost = enter_dead[i] if dest == 2 else (leave_dead[i] if src == 2 else team[i])
                value = scores[src] - cost
                if value > best:
                    best, origin = value, src
            update[dest] = best + e[i, dest]
            back[i, dest] = origin
        scores = update - np.max(update)
    result = np.empty(n, np.int8)
    result[-1] = np.argmax(scores)
    for i in range(n-1, 0, -1):
        result[i-1] = back[i, result[i]]
    return result


@njit(cache=True)
def duration_path(e, team, enter_dead, leave_dead, short_penalty, age_cap,
                  initial_censored=True, final_censored=True):
    """Exact DP for soft cost lambda*max(0,1-duration/age_cap) on completed live runs.

    The capped age has an absorbing continuation, so there is no maximum duration.
    Initial and terminal partial runs can be censored. D always resets team ownership.
    """
    if short_penalty == 0.:
        return path3(e, team, enter_dead, leave_dead)
    n, k = len(e), age_cap
    states, dead = 2*k+1, 2*k
    back = np.zeros((n, states), np.int16)
    scores = np.full(states, -np.inf)
    start_age = k-1 if initial_censored else 0
    scores[start_age], scores[k+start_age], scores[dead] = e[0,0], e[0,1], e[0,2]
    penalty = short_penalty * (1 - np.arange(1,k+1)/k)
    for i in range(1,n):
        update = np.full(states, -np.inf)
        # Stay transitions first, making equality favour continuation.
        for side in range(2):
            for age in range(k-1, -1, -1):
                src = side*k+age
                dest = side*k+min(age+1,k-1)
                value = scores[src] + e[i,side]
                if value > update[dest]:
                    update[dest], back[i,dest] = value, src
        update[dead], back[i,dead] = scores[dead]+e[i,2], dead
        for side in range(2):
            best, src_best = -np.inf, side*k
            for age in range(k):
                value = scores[side*k+age] - penalty[age]
                if value > best:
                    best, src_best = value, side*k+age
            dest = (1-side)*k
            value = best - team[i] + e[i,1-side]
            if value > update[dest]:
                update[dest], back[i,dest] = value, src_best
            value = best - enter_dead[i] + e[i,2]
            if value > update[dead]:
                update[dead], back[i,dead] = value, src_best
            dest = side*k
            value = scores[dead] - leave_dead[i] + e[i,side]
            if value > update[dest]:
                update[dest], back[i,dest] = value, dead
        scores = update - np.max(update)
    if not final_censored:
        for side in range(2):
            for age in range(k):
                scores[side*k+age] -= penalty[age]
    state = np.argmax(scores)
    result = np.empty(n,np.int8)
    for i in range(n-1,-1,-1):
        result[i] = 2 if state == dead else state//k
        if i:
            state = back[i,state]
    return result


@njit(cache=True)
def binary_path(e, to_dead, to_live):
    scores = e[0].astype(np.float64)
    back = np.zeros((len(e),2),np.int8)
    for i in range(1,len(e)):
        update = np.empty(2,np.float64)
        for dest in range(2):
            best, origin = scores[dest],dest
            val = scores[1-dest] - (to_dead[i] if dest == 1 else to_live[i])
            if val > best:
                best,origin=val,1-dest
            update[dest] = best + e[i,dest]
            back[i,dest] = origin
        scores = update - np.max(update)
    out = np.empty(len(e),np.int8)
    out[-1] = np.argmax(scores)
    for i in range(len(e)-1,0,-1):
        out[i-1] = back[i,out[i]]
    return out


def decode(p_live, p_home, seen, period, boundary, mode, team_switch_cost,
           dead_switch_cost, boundary_strength, short_spell_penalty,
           duration_frames, missing_team_temperature=.35):
    n = len(p_live)
    if n == 0 or boundary.shape != (n,3) or not (n==len(p_home)==len(seen)==len(period)):
        raise ValueError("Boundary/emission alignment mismatch")
    if not (np.isfinite(p_live).all() and np.isfinite(p_home).all() and np.isfinite(boundary).all()):
        raise ValueError("Nonfinite prediction")
    if any(np.any((p < 0) | (p > 1)) for p in (p_live, p_home, boundary)):
        raise ValueError("Probabilities must be in [0,1]")
    costs = [team_switch_cost, dead_switch_cost, boundary_strength, short_spell_penalty, missing_team_temperature]
    if not np.isfinite(costs).all() or min(costs) < 0 or not 1 <= duration_frames <= 1000:
        raise ValueError("Invalid decoder parameters")
    if mode not in {"joint", "hierarchical"}:
        raise ValueError("Unknown decoding mode")
    probs = emission_probabilities(p_live,p_home,seen,missing_team_temperature)
    e = np.log(np.clip(probs,1e-8,1.))
    evidence = -np.log(np.clip(boundary,1e-3,1.)) * boundary_strength
    team = team_switch_cost + evidence[:,0]
    enter = dead_switch_cost + evidence[:,1]
    leave = dead_switch_cost + evidence[:,2]
    out = np.full(n,X,np.int8)
    for a,b in blocks(period):
        if period[a] == 0:
            continue
        if mode == "joint":
            out[a:b] = duration_path(e[a:b],team[a:b],enter[a:b],leave[a:b],short_spell_penalty,duration_frames)
        elif mode == "hierarchical":
            live = np.clip(p_live[a:b],1e-5,1-1e-5)
            state = binary_path(np.column_stack([np.log(live),np.log1p(-live)]), enter[a:b],leave[a:b])
            out[a:b] = 2
            edges = np.r_[0,np.flatnonzero(np.diff(state)!=0)+1,len(state)]
            for left,right in zip(edges[:-1],edges[1:]):
                if state[left] != 0:
                    continue
                s,t=a+left,a+right
                owner_e=e[s:t].copy()
                owner_e[:,:2] -= np.log(live[left:right,None])
                owner_e[:,2] = -np.inf
                out[s:t]=duration_path(owner_e,team[s:t],enter[s:t],leave[s:t],short_spell_penalty,duration_frames,
                                       left==0,right==len(state))
        else:
            raise ValueError("Unknown decoding mode")
    return out
