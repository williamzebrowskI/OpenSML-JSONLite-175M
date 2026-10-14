"""Evaluation cadence for short one-pass corpora and future larger collections."""
import math

def evaluation_due(step,total,before,tokens,plan):
    quartiles={math.ceil(total*f) for f in (.25,.5,.75)} if plan['pass_tokens']<plan['eval_every_tokens'] else set()
    return (step==total or step in quartiles or step%plan['eval_every_steps']==0
            or tokens//plan['eval_every_tokens']!=before//plan['eval_every_tokens'])
