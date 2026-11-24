"""Live-trained residual MLP for WeChat Jump Jump. Returns seconds or None."""
from math import isfinite, tanh
LINEAR = [534.0746511220153, 306.9159361384477]
WEIGHTS = [[[-1.5739801902020853, 0.9958615769508367, 1.3074413562238325, -1.2807357910442565, -1.5356444231700006, -0.9849920253420213]], [-0.6438261077384688, 0.02059675515684731, -0.6382841610729058, -0.409757416569455, 0.8458314092725435, 0.048285783497598586], [[-1.443554632897752, -0.32195774502240004, -0.18392050031868512, 1.0442071309770364], [0.6311153128873195, 0.44008437520344923, 0.6823056402350544, -0.7024233386573026], [0.18145202942500308, 0.4762885766028347, 1.3400612570935244, -0.44177571930582926], [-1.055824316638525, -0.40438782950825336, -0.35385057291601457, 0.9240030348486997], [-0.0833166985285688, -0.4393405800880362, -1.6079746524970668, 0.3406741308732495], [-0.5672287710834254, -0.44571018459132467, -0.7373186094518275, 0.6737928085929747]], [1.4827144119526996, -0.42210823508344625, -1.4108817260797897, -0.4169935359128645], [[1.4139151011513298], [1.293681952779917], [-1.1387514522028965], [0.9752584706764852]], [-0.9812736814791294]]
POINTS = ((65,.230), (102,.337), (125,.4189), (170,.5506), (220,.7107), (268,.8429))


def _fixed(distance):
    for (left, a), (right, b) in zip(POINTS, POINTS[1:]):
        if distance <= right:
            return a + (distance-left)*(b-a)/(right-left)
    return POINTS[-1][1] + (distance-POINTS[-1][0])/320


def predict_hold(length):
    """Return blended model time on [85,285] in 543px-window axial pixels, else None."""
    try:
        x=float(length)
    except (TypeError,ValueError):
        return None
    if not isfinite(x) or not 85<=x<=285:
        return None
    base=_fixed(x)
    x_scaled=(x-165)/100
    w1,b1,w2,b2,w3,b3=WEIGHTS
    hidden=[tanh(x_scaled*v+b) for v,b in zip(w1[0],b1)]
    hidden2=[tanh(sum(v*row[j] for v,row in zip(hidden,w2))+b)
             for j,b in enumerate(b2)]
    residual=sum(v*row[0] for v,row in zip(hidden2,w3))+b3[0]
    seconds=(LINEAR[0]+x_scaled*LINEAR[1]+10*residual)/1000
    # Zero derivative in the blend weight at each edge produces a smooth join.
    u=max(0,min(1,(x-85)/12,(285-x)/21))
    weight=u*u*(3-2*u)
    return base+weight*(seconds-base)

# Online training metadata.
MODEL_VERSION = 'online-20261125-033446-274830'
TRAINING_META = {'trained_samples': 651, 'train_measured_through': '2026-11-25T03:34:45.643', 'fallback_version': 'fixed-tail-320-side12-and-score-v2', 'training_run': '20261125_033415_445789', 'training_jump': 15, 'train_started': '2026-11-25T03:34:46.274', 'features': 'camera-corrected axial pixels, 543px reference', 'domain_px': [85, 285]}
