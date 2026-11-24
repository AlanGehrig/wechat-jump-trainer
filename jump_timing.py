"""Live-trained residual MLP for WeChat Jump Jump. Returns seconds or None."""
from math import isfinite, tanh
LINEAR = [534.6683561186779, 306.6271507380311]
WEIGHTS = [[[-1.4650672071653368, 1.0913536495532077, 1.718709395366281, -1.342058029046673, -1.7755825503793028, -0.9526668378314708]], [-0.5385086260804826, 0.28940221323403714, -1.0197991651202674, -0.4664053953425246, 1.065687852553695, -0.15334912928475977], [[-1.184130148845764, 0.9123278203513947, -0.2590705923183818, -0.48718798407895947], [0.8428900011102056, -0.577632723900543, 0.42830555578760693, 0.5427221476768915], [0.11692868010347285, 0.6078738189740148, 1.496887438514272, 0.7194729893742529], [-1.0642331122988873, 0.8099879613608373, -0.3088457652225562, -0.5118187650664319], [-0.10311392272038807, -0.655782881716021, -1.5591365533894568, -0.711040630755444], [-0.704731953912333, 0.4295258726688156, -0.506710089384771, -0.5412268147409988]], [1.438413500107961, 1.7058691563257349, -1.0824353079062703, -0.4076203791649591], [[1.5468268864027193], [1.2518245773608803], [-1.395687202706341], [1.1098022150944717]], [-1.7270309252807117]]
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
MODEL_VERSION = 'online-20261125-025516-206200'
TRAINING_META = {'trained_samples': 186, 'train_measured_through': '2026-11-25T02:55:07.746', 'fallback_version': 'fixed-tail-320-side12-and-score-v2', 'training_run': '20261125_025240_077779', 'training_jump': 70, 'train_started': '2026-11-25T02:55:16.206', 'features': 'camera-corrected axial pixels, 543px reference', 'domain_px': [85, 285]}
