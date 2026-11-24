"""Live-trained residual MLP for WeChat Jump Jump. Returns seconds or None."""
from math import isfinite, tanh
LINEAR = [533.8555868518615, 307.6695098533243]
WEIGHTS = [[[-1.4556996224042844, 1.0827062724667853, 1.3705773673605297, -1.2704727687288335, -1.5698942862939758, -1.0105549271172045]], [-0.5597901318896451, 0.17421833502634992, -0.6973918932542564, -0.398673694989366, 0.8733652495368449, -0.013881255043035483], [[-1.2899666705032502, -0.3756260349885036, -0.24345484353294075, 0.9596663044541138], [0.816926575223109, 0.4517071525845659, 0.5648475220002468, -0.7776955736667581], [0.17333492043271592, 0.46976959837816573, 1.4202687103686578, -0.4375852402911372], [-1.0631670770175887, -0.4233603904312219, -0.36663866790003197, 0.8875192460177793], [-0.089679323538631, -0.42858189007692926, -1.6424666185207277, 0.35555118484589227], [-0.6627987413830765, -0.45814472726964106, -0.6977309446599814, 0.7026238387152087]], [1.4931270237039496, -0.5779178370560467, -1.4694093245138269, -0.38110751599820814], [[1.4336320595312375], [1.2962034552602015], [-1.1961492900256294], [1.0282124910539987]], [-0.9491633287239054]]
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
MODEL_VERSION = 'online-20261125-031818-527194'
TRAINING_META = {'trained_samples': 459, 'train_measured_through': '2026-11-25T03:18:17.977', 'fallback_version': 'fixed-tail-320-side12-and-score-v2', 'training_run': '20261125_031714_251900', 'training_jump': 29, 'train_started': '2026-11-25T03:18:18.527', 'features': 'camera-corrected axial pixels, 543px reference', 'domain_px': [85, 285]}
