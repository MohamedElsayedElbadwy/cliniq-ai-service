"""Deterministic metrics/calibration helpers with only NumPy dependencies."""
import numpy as np


def softmax(logits,temperature=1.):
    values=np.asarray(logits,dtype=np.float64)/temperature
    values-=values.max(axis=-1,keepdims=True)
    scores=np.exp(values)
    return scores/scores.sum(axis=-1,keepdims=True)


def classification_metrics(labels,logits,class_names):
    labels=np.asarray(labels,dtype=int)
    predicted=np.asarray(logits).argmax(axis=1)
    k=len(class_names)
    matrix=np.bincount(labels*k+predicted,minlength=k*k).reshape(k,k)
    tp=np.diag(matrix)
    support=matrix.sum(axis=1)
    precision=np.divide(tp,matrix.sum(axis=0),out=np.zeros(k,dtype=float),where=matrix.sum(axis=0)>0)
    recall=np.divide(tp,support,out=np.zeros(k,dtype=float),where=support>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(k,dtype=float),where=precision+recall>0)
    sorted_indexes=np.argsort(-np.asarray(logits),axis=1)
    return dict(accuracy=float(np.mean(labels==predicted)),macroPrecision=float(precision.mean()),
        macroRecall=float(recall.mean()),macroF1=float(f1.mean()),
        weightedPrecision=float(np.average(precision,weights=support)),weightedRecall=float(np.average(recall,weights=support)),
        weightedF1=float(np.average(f1,weights=support)),
        top3Accuracy=float(np.mean(np.any(sorted_indexes[:,:min(3,k)]==labels[:,None],axis=1))),
        top5Accuracy=float(np.mean(np.any(sorted_indexes[:,:min(5,k)]==labels[:,None],axis=1))),
        perClass={name:dict(precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),support=int(support[i])) for i,name in enumerate(class_names)},
        confusionMatrix=matrix.tolist(),support=int(len(labels)))


def fit_temperature(logits,labels):
    def loss(log_temp):
        values=np.asarray(logits,dtype=np.float64)/np.exp(log_temp)
        values-=values.max(axis=1,keepdims=True)
        return float(np.mean(np.log(np.exp(values).sum(axis=1))-values[np.arange(len(labels)),labels]))
    a,b=np.log(.1),np.log(20.)
    ratio=(np.sqrt(5)-1)/2
    for _ in range(80):
        c,d=b-ratio*(b-a),a+ratio*(b-a)
        if loss(c)<loss(d):
            b=d
        else:
            a=c
    return float(np.exp((a+b)/2))


def calibration_metrics(logits,labels,temperature=1.,bins=15):
    probs=softmax(logits,temperature)
    confidence=probs.max(axis=1)
    correct=probs.argmax(axis=1)==labels
    rows=[]
    ece=0.
    for i in range(bins):
        mask=(confidence>=i/bins)&(confidence<(i+1)/bins if i<bins-1 else confidence<=1)
        count=int(mask.sum())
        accuracy=float(correct[mask].mean()) if count else None
        conf=float(confidence[mask].mean()) if count else None
        if count:
            ece+=count/len(labels)*abs(accuracy-conf)
        rows.append(dict(lower=i/bins,upper=(i+1)/bins,count=count,accuracy=accuracy,confidence=conf))
    return dict(ece=float(ece),bins=rows,
        nll=float(-np.log(np.clip(probs[np.arange(len(labels)),labels],1e-12,1)).mean()))


def choose_abstention(logits,labels,temperature,target_accuracy=.9,min_answered=50):
    probs=softmax(logits,temperature)
    confidence=probs.max(axis=1)
    correct=probs.argmax(axis=1)==labels
    for threshold in sorted(set(confidence.tolist())):
        mask=confidence>=threshold
        if mask.sum()>=min_answered and correct[mask].mean()>=target_accuracy:
            return dict(threshold=float(threshold),validated=True,coverage=float(mask.mean()),
                answered=int(mask.sum()),accuracy=float(correct[mask].mean()),targetAccuracy=target_accuracy,
                caveat='Validation-set empirical accuracy only; not a clinical or OOD safety guarantee.')
    return dict(threshold=0.,validated=False,coverage=1.,answered=len(labels),accuracy=float(correct.mean()),
        targetAccuracy=target_accuracy,caveat='No validation threshold met the target with sufficient support. No validated abstention guarantee.')
