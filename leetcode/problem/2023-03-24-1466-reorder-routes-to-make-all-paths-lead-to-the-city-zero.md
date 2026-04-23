---
layout: leetcode-entry
title: "1466. Reorder Routes to Make All Paths Lead to the City Zero"
permalink: "/leetcode/problem/2023-03-24-1466-reorder-routes-to-make-all-paths-lead-to-the-city-zero/"
leetcode_ui: true
entry_slug: "2023-03-24-1466-reorder-routes-to-make-all-paths-lead-to-the-city-zero"
---

[1466. Reorder Routes to Make All Paths Lead to the City Zero](https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/description/) medium

[blog post](https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/solutions/3334850/kotlin-bfs/)

```kotlin

    fun minReorder(n: Int, connections: Array<IntArray>): Int {
        val edges = mutableMapOf<Int, MutableList<Int>>()
        connections.forEach { (from, to) ->
            edges.getOrPut(from, { mutableListOf() }) += to
            edges.getOrPut(to, { mutableListOf() }) += -from
        }
        val visited = HashSet<Int>()
            var count = 0
            with(ArrayDeque<Int>().apply { add(0) }) {
                fun addNext(x: Int) {
                    if (visited.add(Math.abs(x))) {
                        add(Math.abs(x))
                        if (x > 0) count++
                    }
                }
                while (isNotEmpty()) {
                    repeat(size) {
                        val from = poll()
                        edges[from]?.forEach { addNext(it) }
                        edges[-from]?.forEach { addNext(it) }
                    }
                }
            }
            return count
        }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/158
#### Intuition
If our roads are undirected, the problem is simple: traverse with BFS from `0` and count how many roads are in the opposite direction.

#### Approach
We can use data structure or just use sign to encode the direction.
#### Complexity
- Time complexity:
$$O(V+E)$$
- Space complexity:
$$O(V+E)$$

