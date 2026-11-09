import argparse, json
from xiangqi import XiangqiBoard, XiangqiEngine

def main():
    ap=argparse.ArgumentParser(description="Xiangqi AI move analyser")
    ap.add_argument("--depth",type=int,default=2)
    ap.add_argument("--moves",nargs="*",default=[],help="Moves as four digits, e.g. 6050")
    ap.add_argument("--top",type=int,default=5)
    args=ap.parse_args()
    board=XiangqiBoard()
    for move in args.moves: board.push_uci(move)
    print("Current board (R=Red, B=Black; uppercase=Red):")
    print(board)
    engine=XiangqiEngine()
    result=engine.search_best_move(board,args.depth,args.top)
    print("\\nEngine analysis:")
    print(json.dumps(result,indent=2))
    if result["best_uci"]:
        print(f"\\nSuggested move: {result['best_uci']} (from row,col to row,col)")
        print("Apply with: python main.py --moves", " ".join(args.moves+[result["best_uci"]]))

if __name__=="__main__": main()
