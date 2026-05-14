import pandas as pd
from database.connection import read_table

pred = read_table('predictions')
print('=== PREDICTION DISTRIBUTION ===')
print(f'Total predictions: {len(pred)}')
if 'predicted_trend' in pred.columns:
    up = (pred['predicted_trend'] == 1).sum()
    down = (pred['predicted_trend'] == 0).sum()
    print(f'Predicted UP:   {up} ({up/len(pred)*100:.1f}%)')
    print(f'Predicted DOWN: {down} ({down/len(pred)*100:.1f}%)')

if 'predicted_proba' in pred.columns:
    print(f'\n=== PROBABILITY DISTRIBUTION ===')
    print(pred['predicted_proba'].describe())
    proba = pred['predicted_proba']
    print(f'\nPercentiles:')
    for q in [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]:
        print(f'  {q*100:.0f}th: {proba.quantile(q):.4f}')

if 'actual_trend' in pred.columns:
    known = pred.dropna(subset=['actual_trend'])
    print(f'\n=== ACTUAL vs PREDICTED (known: {len(known)}) ===')
    actual_up = (known['actual_trend'] == 1).sum()
    actual_down = (known['actual_trend'] == 0).sum()
    print(f'Actual UP:   {actual_up} ({actual_up/len(known)*100:.1f}%)')
    print(f'Actual DOWN: {actual_down} ({actual_down/len(known)*100:.1f}%)')

    tp = ((known['predicted_trend']==1) & (known['actual_trend']==1)).sum()
    fp = ((known['predicted_trend']==1) & (known['actual_trend']==0)).sum()
    tn = ((known['predicted_trend']==0) & (known['actual_trend']==0)).sum()
    fn = ((known['predicted_trend']==0) & (known['actual_trend']==1)).sum()
    print(f'\nConfusion Matrix:')
    print(f'  TP (predict UP, actual UP):     {tp}')
    print(f'  FP (predict UP, actual DOWN):   {fp}  <-- FALSE POSITIVE')
    print(f'  TN (predict DOWN, actual DOWN): {tn}')
    print(f'  FN (predict DOWN, actual UP):   {fn}')
    if tp+fp > 0:
        print(f'\nPrecision UP:   {tp/(tp+fp):.3f}')
    if tp+fn > 0:
        print(f'Recall UP:      {tp/(tp+fn):.3f}')
    if tn+fn > 0:
        print(f'Precision DOWN: {tn/(tn+fn):.3f}')
    if tn+fp > 0:
        print(f'Recall DOWN:    {tn/(tn+fp):.3f}')

    # Test different thresholds
    print('\n=== OPTIMAL THRESHOLD SEARCH ===')
    print(f'{"Thresh":>7} {"Acc":>6} {"Prec_UP":>8} {"Rec_UP":>7} {"Prec_DN":>8} {"Rec_DN":>7} {"F1":>6} {"UP%":>5}')
    for t in [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
        p = (known['predicted_proba'] >= t).astype(int)
        a = known['actual_trend'].astype(int)
        _tp = ((p==1) & (a==1)).sum()
        _fp = ((p==1) & (a==0)).sum()
        _tn = ((p==0) & (a==0)).sum()
        _fn = ((p==0) & (a==1)).sum()
        acc = (_tp+_tn) / len(known) * 100
        prec_up = _tp/(_tp+_fp) if _tp+_fp > 0 else 0
        rec_up = _tp/(_tp+_fn) if _tp+_fn > 0 else 0
        prec_dn = _tn/(_tn+_fn) if _tn+_fn > 0 else 0
        rec_dn = _tn/(_tn+_fp) if _tn+_fp > 0 else 0
        f1 = 2*prec_up*rec_up/(prec_up+rec_up) if prec_up+rec_up > 0 else 0
        up_pct = p.sum() / len(p) * 100
        print(f'{t:>7.2f} {acc:>6.1f} {prec_up:>8.3f} {rec_up:>7.3f} {prec_dn:>8.3f} {rec_dn:>7.3f} {f1:>6.3f} {up_pct:>5.1f}')
