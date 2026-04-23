---
layout: leetcode-entry
title: "934. Shortest Bridge"
permalink: "/leetcode/problem/2023-05-21-934-shortest-bridge/"
leetcode_ui: true
entry_slug: "2023-05-21-934-shortest-bridge"
---

[934. Shortest Bridge](https://leetcode.com/problems/shortest-bridge/description/) medium
[blog post](https://leetcode.com/problems/shortest-bridge/solutions/3546914/kotlin-dfs-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/21052023-934-shortest-bridge?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/219
#### Problem TLDR
Find the shortest path from one island of `1`'s to another.
#### Intuition
Shortest path can be found with Breadth-First Search if we start it from every border cell of the one of the islands.
To detect border cell, we have to make separate DFS.

#### Approach
* modify grid to store `visited` set
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

fun Array<IntArray>.inRange(xy: Pair<Int, Int>) = (0..lastIndex).let {
    xy.first in it && xy.second in it
}
fun Pair<Int, Int>.siblings() = arrayOf(
(first - 1) to second, first to (second - 1),
(first + 1) to second, first to (second + 1)
)
fun shortestBridge(grid: Array<IntArray>): Int {
    val queue = ArrayDeque<Pair<Int, Int>>()
    fun dfs(x: Int, y: Int) {
        if (grid[y][x] == 1) {
            grid[y][x] = 2
            (x to y).siblings().filter { grid.inRange(it) }.forEach { dfs(it.first, it.second) }
        } else if (grid[y][x] == 0) queue.add(x to y)
    }
    (0 until grid.size * grid.size)
    .map { it / grid.size to it % grid.size}
    .filter { (y, x) -> grid[y][x] == 1 }
    .first().let { (y, x) -> dfs(x, y)}
    return with (queue) {
        var steps = 1
        while (isNotEmpty()) {
            repeat(size) {
                val xy = poll()
                if (grid.inRange(xy)) {
                    val (x, y) = xy
                    if (grid[y][x] == 1) return@shortestBridge steps - 1
                    if (grid[y][x] == 0) {
                        grid[y][x] = 3
                        addAll(xy.siblings().filter { grid.inRange(it) })
                    }
                }
            }
            steps++
        }
        -1
    }
}

```

