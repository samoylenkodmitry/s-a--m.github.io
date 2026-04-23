---
layout: leetcode-entry
title: "1162. As Far from Land as Possible"
permalink: "/leetcode/problem/2023-02-10-1162-as-far-from-land-as-possible/"
leetcode_ui: true
entry_slug: "2023-02-10-1162-as-far-from-land-as-possible"
---

[1162. As Far from Land as Possible](https://leetcode.com/problems/as-far-from-land-as-possible/description/) medium

[blog post](https://leetcode.com/problems/as-far-from-land-as-possible/solutions/3167082/kotlin-bfs/)

```kotlin
    fun maxDistance(grid: Array<IntArray>): Int = with(ArrayDeque<Pair<Int, Int>>()) {
        val n = grid.size
        val visited = hashSetOf<Pair<Int, Int>>()
        fun tryAdd(x: Int, y: Int) {
            if (x < 0 || y < 0 || x >= n || y >= n) return
            (x to y).let { if (visited.add(it)) add(it) }
        }
        for (yStart in 0 until n)
            for (xStart in 0 until n)
                if (grid[yStart][xStart] == 1) tryAdd(xStart, yStart)
        if (size == n*n) return -1
        var dist = -1
        while(isNotEmpty()) {
            repeat(size) {
                val (x, y) = poll()
                tryAdd(x-1, y)
                tryAdd(x, y-1)
                tryAdd(x+1, y)
                tryAdd(x, y+1)
            }
            dist++
        }
        dist
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/114
#### Intuition
Let's do a wave from each land and wait until all the last water cell reached. This cell will be the answer.
#### Approach
Add all land cells into BFS, then just run it.
#### Complexity
- Time complexity:
  $$O(n^2)$$
- Space complexity:
  $$O(n^2)$$

