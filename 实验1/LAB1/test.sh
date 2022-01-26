if [[ $(grep 'import' myImpl.py) != 'import util' ]]; then
  echo "Wrong import"
  exit 1
fi

cd search
python autograder.py -q q1
python pacman.py -l mediumMaze -p SearchAgent --frameTime 0
python autograder.py -q q2
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs --frameTime 0
python autograder.py -q q3
python pacman.py -l mediumMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic --frameTime 0
cd ../multiagent
python autograder.py -q q2 --no-graphics
python autograder.py -q q3 --no-graphics
python pacman.py -p AlphaBetaAgent -l mediumClassic --frameTime 0

