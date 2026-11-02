"""Root-cause variable mapping for TEP faults (0-based column indices).

Source: Downs & Vogel (1993), Table of process disturbances (IDV) and the
measured/manipulated variable list. A fault is mapped ONLY when its disturbance
acts on a specific stream or loop whose flow/temperature/valve is a listed
variable. Faults with diffuse effects (IDV 3, 8-10, 13, 16-21) are excluded.
1-based names: XMEAS(k) -> column k-1, XMV(k) -> column 40+k.
"""
def xmeas(k): return k - 1
def xmv(k): return 40 + k

ROOT_CAUSES = {
    1: {xmeas(4), xmv(4)},            # A/C feed ratio, stream 4 -> A and C feed flow, its valve
    2: {xmeas(4), xmv(4)},            # B composition, stream 4
    4: {xmeas(9), xmeas(21), xmv(10)},# reactor cooling water inlet temp -> reactor temp, CW outlet temp, CW flow valve
    5: {xmeas(22), xmv(11)},          # condenser cooling water inlet temp -> separator CW outlet temp, condenser CW valve
    6: {xmeas(1), xmv(3)},            # A feed loss (stream 1) -> A feed flow, A feed valve
    7: {xmeas(4), xmv(4)},            # C header pressure loss (stream 4)
    11: {xmeas(9), xmeas(21), xmv(10)},
    12: {xmeas(22), xmv(11)},
    14: {xmeas(9), xmeas(21), xmv(10)},# reactor cooling water valve sticking
    15: {xmeas(22), xmv(11)},          # condenser cooling water valve sticking
}

# Valve -> controlled-flow pairs listed together in the Downs & Vogel variable tables
KNOWN_EDGES = [
    (xmv(1), xmeas(2)), (xmv(2), xmeas(3)), (xmv(3), xmeas(1)), (xmv(4), xmeas(4)),
    (xmv(5), xmeas(5)), (xmv(6), xmeas(10)), (xmv(7), xmeas(14)), (xmv(8), xmeas(17)),
    (xmv(9), xmeas(19)), (xmv(10), xmeas(9)), (xmv(10), xmeas(21)), (xmv(11), xmeas(22)),
]
