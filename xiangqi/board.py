from dataclasses import dataclass
from typing import Optional, List, Tuple

# Board coordinates: rows 0..9 (Black at top), columns 0..8.
# Uppercase = Red, lowercase = Black.
INITIAL = [
    list("rheakaehr"),
    list("........."),
    list(".c.....c."),
    list("p.p.p.p.p"),
    list("........."),
    list("........."),
    list("P.P.P.P.P"),
    list(".C.....C."),
    list("........."),
    list("RHEAKAEHR"),
]
# Piece letters: K king, A advisor, E elephant, H horse, R rook, C cannon, P pawn.
VALUES = {"K":10000,"A":200,"E":200,"H":450,"R":900,"C":450,"P":100}

@dataclass(frozen=True)
class Move:
    fr: int; fc: int; tr: int; tc: int
    def uci(self): return f"{self.fr}{self.fc}{self.tr}{self.tc}"
    def __str__(self): return self.uci()

class XiangqiBoard:
    def __init__(self, grid=None, turn="R"):
        self.grid = [[("" if p=="." else p) for p in row] for row in (grid or INITIAL)]
        self.turn = turn
        self.history = []

    def copy(self):
        b = XiangqiBoard(self.grid, self.turn)
        b.history = self.history[:]
        return b

    def inside(self,r,c): return 0 <= r < 10 and 0 <= c < 9
    def side(self,p): return "R" if p.isupper() else "B"
    def palace(self,side,r,c):
        return 7 <= r <= 9 and 3 <= c <= 5 if side=="R" else 0 <= r <= 2 and 3 <= c <= 5
    def crossed_river(self,side,r): return r <= 4 if side=="R" else r >= 5
    def _add(self,moves,r,c,nr,nc,side):
        if self.inside(nr,nc):
            q=self.grid[nr][nc]
            if not q or self.side(q)!=side: moves.append(Move(r,c,nr,nc))

    def pseudo_moves(self,r,c):
        p=self.grid[r][c]
        if not p: return []
        side=self.side(p); kind=p.upper(); out=[]
        def add(nr,nc): self._add(out,r,c,nr,nc,side)
        if kind=="K":
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nr,nc=r+dr,c+dc
                if self.palace(side,nr,nc): add(nr,nc)
            # Flying general capture
            step=-1 if side=="R" else 1
            nr=r+step
            while self.inside(nr,c):
                if self.grid[nr][c]:
                    if self.grid[nr][c].upper()=="K" and self.side(self.grid[nr][c])!=side:
                        out.append(Move(r,c,nr,c))
                    break
                nr+=step
        elif kind=="A":
            for dr,dc in ((1,1),(1,-1),(-1,1),(-1,-1)):
                nr,nc=r+dr,c+dc
                if self.palace(side,nr,nc): add(nr,nc)
        elif kind=="E":
            for dr,dc in ((2,2),(2,-2),(-2,2),(-2,-2)):
                nr,nc=r+dr,c+dc; mr,mc=r+dr//2,c+dc//2
                own_half = nr>=5 if side=="R" else nr<=4
                if self.inside(nr,nc) and own_half and not self.grid[mr][mc]: add(nr,nc)
        elif kind=="H":
            for dr,dc,lr,lc in ((2,1,1,0),(2,-1,1,0),(-2,1,-1,0),(-2,-1,-1,0),(1,2,0,1),(-1,2,0,1),(1,-2,0,-1),(-1,-2,0,-1)):
                nr,nc=r+dr,c+dc
                if self.inside(nr,nc) and not self.grid[r+lr][c+lc]: add(nr,nc)
        elif kind in ("R","C"):
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nr,nc=r+dr,c+dc; screened=False
                while self.inside(nr,nc):
                    q=self.grid[nr][nc]
                    if kind=="R":
                        if not q: out.append(Move(r,c,nr,nc))
                        else:
                            if self.side(q)!=side: out.append(Move(r,c,nr,nc))
                            break
                    else:
                        if not screened:
                            if not q: out.append(Move(r,c,nr,nc))
                            else: screened=True
                        elif q:
                            if self.side(q)!=side: out.append(Move(r,c,nr,nc))
                            break
                    nr+=dr; nc+=dc
        elif kind=="P":
            dr=-1 if side=="R" else 1
            add(r+dr,c)
            if self.crossed_river(side,r):
                add(r,c-1); add(r,c+1)
        return out

    def make(self,m):
        p=self.grid[m.fr][m.fc]; captured=self.grid[m.tr][m.tc]
        self.history.append((m,p,captured,self.turn))
        self.grid[m.tr][m.tc]=p; self.grid[m.fr][m.fc]=""
        self.turn="B" if self.turn=="R" else "R"
    def undo(self):
        m,p,captured,turn=self.history.pop()
        self.grid[m.fr][m.fc]=p; self.grid[m.tr][m.tc]=captured; self.turn=turn
    def king_pos(self,side):
        target="K" if side=="R" else "k"
        for r in range(10):
            for c in range(9):
                if self.grid[r][c]==target: return r,c
        return None
    def in_check(self,side):
        kp=self.king_pos(side)
        if kp is None: return True
        enemy="B" if side=="R" else "R"
        for r in range(10):
            for c in range(9):
                p=self.grid[r][c]
                if p and self.side(p)==enemy:
                    if kp in [(m.tr,m.tc) for m in self.pseudo_moves(r,c)]: return True
        return False
    def get_legal_moves(self):
        out=[]
        for r in range(10):
            for c in range(9):
                p=self.grid[r][c]
                if p and self.side(p)==self.turn:
                    for m in self.pseudo_moves(r,c):
                        self.make(m)
                        safe=not self.in_check("B" if self.turn=="R" else "R")
                        self.undo()
                        if safe: out.append(m)
        return out
    def push_uci(self,s):
        m=next((m for m in self.get_legal_moves() if m.uci()==s),None)
        if m is None: raise ValueError(f"Illegal move: {s}")
        self.make(m)
    def to_tensor(self):
        import numpy as np
        order="KAEHRCPkaehrcp"
        x=np.zeros((14,10,9),dtype="float32")
        for r in range(10):
            for c in range(9):
                p=self.grid[r][c]
                if p in order: x[order.index(p),r,c]=1
        return x
    def __str__(self):
        return "\\n".join(" ".join(p or "." for p in row) for row in self.grid)+f"\\nTurn: {self.turn}"
