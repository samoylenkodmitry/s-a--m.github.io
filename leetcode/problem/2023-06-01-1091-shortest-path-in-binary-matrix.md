---
layout: leetcode-entry
title: "1091. Shortest Path in Binary Matrix"
permalink: "/leetcode/problem/2023-06-01-1091-shortest-path-in-binary-matrix/"
leetcode_ui: true
entry_slug: "2023-06-01-1091-shortest-path-in-binary-matrix"
---

[1091. Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/description/) medium
[blog post](https://leetcode.com/problems/shortest-path-in-binary-matrix/solutions/3584350/kotln-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/01062023-1091-shortest-path-in-binary?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/232
#### Problem TLDR
`0` path length in a binary square matrix.
#### Intuition
Just do BFS.

#### Approach
Some tricks for cleaner code:
* check for x, y in `range`
* iterate over `dirs`. This is a sequence of `x` and `y`
* modify the input array. But don't do this in production code.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun shortestPathBinaryMatrix(grid: Array<IntArray>): Int =
    with(ArrayDeque<Pair<Int, Int>>()) {
        val range = 0..grid.lastIndex
        val dirs = arrayOf(0, 1, 0, -1, -1, 1, 1, -1)
        if (grid[0][0] == 0) add(0 to 0)
        grid[0][0] = -1
        var step = 0
        while (isNotEmpty()) {
            step++
            repeat(size) {
                val (x, y) = poll()
                if (x == grid.lastIndex && y == grid.lastIndex) return step
                var dx = -1
                for (dy in dirs) {
                    val nx = x + dx
                    val ny = y + dy
                    if (nx in range && ny in range && grid[ny][nx] == 0) {
                        grid[ny][nx] = -1
                        add(nx to ny)
                    }
                    dx = dy
                }
            }
        }
        -1
    }

```

