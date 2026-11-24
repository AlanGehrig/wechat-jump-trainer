"""Conservative early-observation gate. This module does not capture or send input."""


def ready_pair(before_score, landed_kind, first, second):
    """Return True for two independently validated ordinary-board observations.

    first/second are (elapsed_seconds, hud_score, scene). The caller captures
    near 1.00/1.12 s after release and supplies only scenes whose pawn, next
    platform, and input geometry passed its normal checks. Use the second
    observation for aiming on success. Otherwise keep the existing 1.30 s path.
    """
    if landed_kind is not None or before_score is None or first is None or second is None:
        return False
    t1, s1, scene1 = first
    t2, s2, scene2 = second
    if not (.98 <= t1 <= 1.07 and 1.10 <= t2 <= 1.20 and t2-t1 >= .075):
        return False
    if s1 is None or s1 <= before_score or s2 != s1:
        return False
    if not scene1 or not scene2:
        return False
    f1, t1xy = scene1.get("foot"), scene1.get("target")
    f2, t2xy = scene2.get("foot"), scene2.get("target")
    if f1 is None or t1xy is None or f2 is None or t2xy is None:
        return False
    return all(abs((t1xy[i]-f1[i]) - (t2xy[i]-f2[i])) <= 1 for i in (0,1))
