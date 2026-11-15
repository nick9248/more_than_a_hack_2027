"""Keep the last team during exactly neutral evidence when paths have equal scores.

This is a deterministic football-rule tie convention, not another learned
confidence threshold. Nonneutral evidence, D, half boundaries, and all strict
preferences of the frozen structured model retain their original meaning.
"""
import numpy as np
from .data import blocks
from .structured import decode as boundary_decode


def duration_cost(labels, cap, weight):
    if not weight:
        return 0.
    edges=np.r_[0,np.flatnonzero(np.diff(labels)!=0)+1,len(labels)]
    start,stop=edges[1:-2],edges[2:-1]
    live=labels[start]<2
    return float(weight*np.maximum(0.,1-(stop[live]-start[live])/cap).sum())


def transition_cost(labels, team, enter, leave, indices):
    previous,current=labels[indices-1],labels[indices]
    cost=np.where(current==2,enter[indices],np.where(previous==2,leave[indices],team[indices]))
    return float(cost[previous!=current].sum())


def retain_neutral_ties(labels,p_home,period,boundary,parameters):
    if parameters["mode"]!="joint":
        raise ValueError("The correction is defined for the selected joint decoder only")
    out=labels.copy()
    evidence=-np.log(np.clip(boundary,1e-3,1.))*parameters["boundary_strength"]
    team=parameters["team_switch_cost"]+evidence[:,0]
    enter=parameters["dead_switch_cost"]+evidence[:,1]
    leave=parameters["dead_switch_cost"]+evidence[:,2]
    for first,last in blocks(period):
        if period[first]==0:
            continue
        z=out[first:last]
        neutral=(p_home[first:last]==.5)&(z<2)
        edges=np.r_[0,np.flatnonzero(np.diff(neutral)!=0)+1,len(z)]
        for a,b in zip(edges[:-1],edges[1:]):
            if not neutral[a] or a==0 or z[a-1]>=2 or np.all(z[a:b]==z[a-1]):
                continue
            trial=z.copy()
            trial[a:b]=z[a-1]
            indices=np.arange(max(1,a),min(len(z),b+1))
            old_cost=transition_cost(z,team[first:last],enter[first:last],leave[first:last],indices)
            new_cost=transition_cost(trial,team[first:last],enter[first:last],leave[first:last],indices)
            old_cost+=duration_cost(z,parameters["duration_frames"],parameters["short_spell_penalty"])
            new_cost+=duration_cost(trial,parameters["duration_frames"],parameters["short_spell_penalty"])
            # Emissions are identical for H/A on exactly neutral frames. Keep the
            # same global objective, allowing only ordinary floating-point roundoff.
            roundoff=64*np.finfo(np.float64).eps*max(1.,abs(old_cost),abs(new_cost))
            if abs(old_cost-new_cost)<=roundoff:
                z[a:b]=trial[a:b]
    return out


def decode(p_live,p_home,seen,period,boundary,**parameters):
    labels=boundary_decode(p_live,p_home,seen,period,boundary,**parameters)
    return retain_neutral_ties(labels,p_home,period,boundary,parameters)
