# Extracted from the user-provided snapshot for regression testing only.
import numpy as np
T_OPEN, T_CLOSE, NEG = 1000, 1001, -1e9

class OGeo:
    """One buffer per micro: clean sequence + N*G blocks [<T>, L slots, </T>, K branch rows] (inserted geometry, re-read)."""

    def __init__(self, T, N, G, L, K):
        self.T, self.N, self.G, self.L, self.K = T, N, G, L, K
        self.PB = (L + 2) + K
        self.S = T + N * G * self.PB
        a = np.arange(N)[:, None]; g = np.arange(G)[None, :]
        self.B = T + (a * G + g) * self.PB                                   # [N, G] block starts
        self.RC = self.B + L + 1                                             # [N, G] closer rows
        self.BR = self.RC[:, :, None] + 1 + np.arange(K)[None, None, :]      # [N, G, K] branch rows
        self.TH = self.B[:, :, None] + np.arange(L + 1)[None, None, :]       # [N, G, L+1] <T> + L slots

    def buffer(self, x, anchors, Z, lens):
        """x [T], anchors [N], Z [G, N, L], lens [G, N] -> buf, pos, mask, tgt, thought-row mask [N, G, L+1]"""
        T, N, G, L, K, S = self.T, self.N, self.G, self.L, self.K, self.S
        buf = np.full(S, T_CLOSE, dtype=np.int64); pos = np.zeros(S, dtype=np.float32)
        m = np.full((S, S), NEG, dtype=np.float32); tgt = np.zeros(S, dtype=np.int64)
        buf[:T] = x; pos[:T] = np.arange(T); m[:T, :T] = np.triu(np.full((T, T), NEG, dtype=np.float32), k=1)
        tgt[:T - 1] = x[1:]
        thm = np.zeros((N, G, L + 1), dtype=np.float32)
        tri = np.triu(np.full((L + 1, L + 1), NEG, dtype=np.float32), k=1)
        slot = np.arange(L + 1, dtype=np.float32)                       # <T> at n+1, Z[j] at n+2+j (plain inserted text)
        for a in range(N):
            n = int(anchors[a])
            for g in range(G):
                B = int(self.B[a, g]); k = max(int(lens[g, a]), 1)
                rows = np.arange(B, B + L + 1)
                buf[B] = T_OPEN; buf[B + 1:B + L + 1] = Z[g, a]
                pos[rows] = n + 1.0 + slot
                m[rows, :n + 1] = 0.0; m[rows[:, None], rows[None, :]] = tri
                tgt[B:B + k] = Z[g, a, :k]; tgt[B + k] = T_CLOSE; thm[a, g, :k + 1] = 1.0
                rc = B + L + 1; buf[rc] = T_CLOSE; pos[rc] = n + 2.0 + k
                m[rc, :n + 1] = 0.0; m[rc, B:B + k + 1] = 0.0; m[rc, rc] = 0.0
                for j in range(K):
                    rt = rc + 1 + j; buf[rt] = x[n + j]; pos[rt] = n + 3.0 + k + j; tgt[rt] = x[n + j + 1]
                    m[rt, :n + 1] = 0.0; m[rt, B:B + k + 1] = 0.0; m[rt, rc:rt + 1] = 0.0
        return buf, pos, m, tgt, thm
