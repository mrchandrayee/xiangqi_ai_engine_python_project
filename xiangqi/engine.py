import math
from .board import VALUES, XiangqiBoard

class XiangqiEngine:
    """Alpha-beta Xiangqi engine with material/positional heuristic evaluation."""
    def __init__(self, model=None, device="cpu"):
        self.model=model
        self.device=device
        if model is not None:
            model.to(device); model.eval()

    def heuristic_eval(self, board):
        score=0
        for r in range(10):
            for c in range(9):
                p=board.grid[r][c]
                if not p: continue
                side=board.side(p); kind=p.upper()
                val=VALUES[kind]
                # Centralization and advancement provide modest positional guidance.
                center=4-abs(4-c)
                if kind in ("H","C","R"): val += center*7
                if kind=="P":
                    progress=(6-r) if side=="R" else (r-3)
                    val += max(0,progress)*12 + (center*3 if board.crossed_river(side,r) else 0)
                score += val if side=="R" else -val
        return score

    def evaluate_state(self, board):
        h=math.tanh(self.heuristic_eval(board)/2500.0)
        if self.model is None: return (0.5+0.5*math.tanh(h*2),0.0,0.5-0.5*math.tanh(h*2)),h
        try:
            import torch
            with torch.no_grad():
                x=torch.from_numpy(board.to_tensor()).unsqueeze(0).to(self.device)
                out=self.model(x)
                if isinstance(out,(tuple,list)): out=out[0]
                probs=torch.softmax(out.reshape(-1)[:3],dim=0).cpu().tolist()
            return tuple(probs),h
        except Exception:
            return (0.5,0.0,0.5),h

    def _ordered(self,board,moves):
        def key(m):
            target=board.grid[m.tr][m.tc]
            return VALUES.get(target.upper(),0) if target else 0
        return sorted(moves,key=key,reverse=True)

    def alpha_beta_search(self,board,depth,alpha=-math.inf,beta=math.inf,maximizing=None):
        if maximizing is None: maximizing=(board.turn=="R")
        legal=board.get_legal_moves()
        if depth<=0 or not legal:
            if not legal:
                score=-9000 if maximizing else 9000
                return score,None
            return self.heuristic_eval(board),None
        best=None
        if maximizing:
            value=-math.inf
            for m in self._ordered(board,legal):
                board.make(m); score,_=self.alpha_beta_search(board,depth-1,alpha,beta,False); board.undo()
                if score>value: value,best=score,m
                alpha=max(alpha,value)
                if beta<=alpha: break
        else:
            value=math.inf
            for m in self._ordered(board,legal):
                board.make(m); score,_=self.alpha_beta_search(board,depth-1,alpha,beta,True); board.undo()
                if score<value: value,best=score,m
                beta=min(beta,value)
                if beta<=alpha: break
        return value,best

    def search_best_move(self,board,depth=2,top_n=5):
        legal=board.get_legal_moves()
        if not legal:
            probs,score=self.evaluate_state(board)
            return {"best_move":None,"best_uci":None,"eval_score":score,"win_probs":probs,"candidates":[]}
        is_red=board.turn=="R"; evaluated=[]
        for m in legal:
            board.make(m)
            score,_=self.alpha_beta_search(board,max(0,depth-1),-math.inf,math.inf,not is_red)
            board.undo()
            relative=score if is_red else -score
            probs,heur=self.evaluate_state_after_move(board,m)
            evaluated.append({"move":str(m),"uci":m.uci(),"piece":board.grid[m.fr][m.fc],"score":round(relative,2),"raw_score":round(score,2),"win_pct":round(probs[0 if is_red else 2]*100,1)})
        evaluated.sort(key=lambda x:x["score"],reverse=True)
        best=evaluated[0]
        return {"best_move":best["move"],"best_uci":best["uci"],"eval_score":best["score"],"win_probs":self.evaluate_state(board)[0],"candidates":evaluated[:top_n]}

    def evaluate_state_after_move(self,board,move):
        board.make(move); result=self.evaluate_state(board); board.undo(); return result
