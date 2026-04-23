---
layout: leetcode-entry
title: "Nearest Exit From Entrance In Maze"
permalink: "/leetcode/problem/2022-11-21-nearest-exit-from-entrance-in-maze/"
leetcode_ui: true
entry_slug: "2022-11-21-nearest-exit-from-entrance-in-maze"
---

[https://leetcode.com/problems/nearest-exit-from-entrance-in-maze/](https://leetcode.com/problems/nearest-exit-from-entrance-in-maze/) medium

```

    fun nearestExit(maze: Array<CharArray>, entrance: IntArray): Int {
        val queue = ArrayDeque<Pair<Int, Int>>()
        queue.add(entrance[1] to entrance[0])
        maze[entrance[0]][entrance[1]] = 'x'
        var steps = 1
        val directions = intArrayOf(-1, 0, 1, 0, -1)
        while(queue.isNotEmpty()) {
            repeat(queue.size){
                val (x, y) = queue.poll()
                for (i in 1..directions.lastIndex) {
                    val nx = x + directions[i-1]
                    val ny = y + directions[i]
                    if (nx in 0..maze[0].lastIndex &&
                            ny in 0..maze.lastIndex &&
                            maze[ny][nx] == '.') {
                        if (nx == 0 ||
                                ny == 0 ||
                                nx == maze[0].lastIndex ||
                                ny == maze.lastIndex) return steps
                        maze[ny][nx] = 'x'
                        queue.add(nx to ny)
                    }
                }
            }
            steps++
        }

        return -1
    }

```

Just do BFS.
* we can modify input matrix, so we can use it as visited array

Complexity: O(N), N - number of cells in maze
Memory: O(N)

