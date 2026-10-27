import heapq


graph = {
    'A': [('B', 2), ('C', 5)],
    'B': [('A', 2), ('D', 1)],
    'C': [('A', 5), ('D', 3)],
    'D': [('B', 1), ('C', 3)]
}

def uniform_cost_search(graph, start, goal):
queue = [(0, start, [start])]

  
priority queue visited;
visited = {}
    while priority_queue:
        cost, current, path = heapq.heappop(priority_queue)

        
        if current == goal:
            return path, cost

       
        if current in visited and visited[current] <= cost:
            continue

        visited[current] = cost

        for neighbor, edge_cost in graph[current]:
            new_cost = cost + edge_cost
            new_path = path + [neighbor]

            heapq.heappush(
                priority_queue,
                (new_cost, neighbor, new_path)
            )