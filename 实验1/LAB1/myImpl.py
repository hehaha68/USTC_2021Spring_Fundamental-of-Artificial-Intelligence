import util

"""
Data sturctures we will use are stack, queue and priority queue.

Stack: first in last out
Queue: first in first out
    collection.push(element): insert element
    element = collection.pop() get and remove element from collection

Priority queue:
    pq.update('eat', 2)
    pq.update('study', 1)
    pq.update('sleep', 3)
pq.pop() will return 'study' because it has highest priority 1.

"""

"""
problem is a object has 3 methods related to search state:

problem.getStartState()
Returns the start state for the search problem.

problem.isGoalState(state)
Returns True if and only if the state is a valid goal state.

problem.getChildren(state)
For a given state, this should return a list of tuples, (next_state,
step_cost), where 'next_state' is a child to the current state, 
and 'step_cost' is the incremental cost of expanding to that child.

"""


def myDepthFirstSearch(problem):
    visited = {}
    frontier = util.Stack()

    frontier.push((problem.getStartState(), None))

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]

        if state not in visited:
            visited[state] = prev_state

            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))

    return []


def myBreadthFirstSearch(problem):
    # YOUR CODE HERE
    visited = {}
    frontier = util.Queue()

    frontier.push((problem.getStartState(), None))

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]

        if state not in visited:
            visited[state] = prev_state

            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))

    return []


def myAStarSearch(problem, heuristic):
    # YOUR CODE HERE
    visited = {}
    cost = {}
    frontier = util.PriorityQueue()

    frontier.update((problem.getStartState(), None), heuristic(problem.getStartState()))
    cost[problem.getStartState()]=0
    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]

        if state not in visited:
            visited[state] = prev_state

            for next_state, step_cost in problem.getChildren(state):
                cost[next_state] = cost[state] + step_cost
                frontier.update((next_state, state), cost[next_state] + heuristic(next_state))

    return []


"""
Game state has 4 methods we can use.

state.isTerminated()
Return True if the state is terminated. We should not continue to search if the state is terminated.

state.isMe()
Return True if it's time for the desired agent to take action. We should check this function to determine whether an agent should maximum or minimum the score.

state.getChildren()
Returns a list of legal state after an agent takes an action.

state.evaluateScore()
Return the score of the state. We should maximum the score for the desired agent.

"""


class MyMinimaxAgent():

    def __init__(self, depth):
        self.depth = depth

    def minimax(self, state, depth):
        if state.isTerminated():
            return None, state.evaluateScore()

        best_state, best_score = None, -float('inf') if state.isMe() else float('inf')

        for child in state.getChildren():
            # YOUR CODE HERE
            temp_depth = depth

            if child.isMe():
                temp_depth = temp_depth - 1
                if temp_depth < 1:
                    child_score = child.evaluateScore()
                else:
                    child_state, child_score = self.minimax(child, temp_depth)

                if best_score > child_score:
                    best_score = child_score
                    best_state = child
            else:
                child_state, child_score = self.minimax(child, depth)
                if state.isMe() and best_score < child_score:
                    best_score = child_score
                    best_state = child
                elif not state.isMe() and best_score > child_score:
                    best_score = child_score
                    best_state = child

        return best_state, best_score

    def getNextState(self, state):
        best_state, _ = self.minimax(state, self.depth)
        return best_state


class MyAlphaBetaAgent():

    def __init__(self, depth):
        self.depth = depth

    def getNextState(self, state):

        def max_value(state, alpha, beta, depth):
            depth = depth - 1

            if state.isTerminated() or depth < 1:
                return state.evaluateScore()
            v = float('-Inf')

            for child in state.getChildren():
                temp_v = min_value(child, alpha, beta, depth)
                if v <= temp_v:
                    v = temp_v
                # 若已经比beta要大了 就没有搜索下去的必要了
                if v > beta:
                    return v
                # 更新alpha的值
                alpha = max(alpha, v)
            return v

        def min_value(state, alpha, beta, depth):

            if state.isTerminated():
                return state.evaluateScore()

            v = float('Inf')

            for child in state.getChildren():
                if child.isMe():
                    temp_v = max_value(child, alpha, beta, depth)
                    if v >= temp_v:
                        v = temp_v
                elif not child.isMe():
                    # 继续下一个Ghost
                    temp_v = min_value(child, alpha, beta, depth)
                    if v >= temp_v:
                        v = temp_v
                # 若比alpha还要小了 就没搜索的必要了
                if v < alpha:
                    return v
                # 更新beta的值
                beta = min(beta, v)
            return v

        alpha = -float('inf')
        beta = float('inf')
        depth = self.depth
        best_state, best_score = None, -float('inf')

        for child in state.getChildren():
            child_score = min_value(child, alpha, beta, depth)
            if best_score < child_score:
                best_score = child_score
                best_state = child
            alpha = max(alpha, best_score)

        return best_state