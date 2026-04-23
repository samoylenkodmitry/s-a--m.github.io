---
layout: leetcode-entry
title: "1443. Minimum Time to Collect All Apples in a Tree"
permalink: "/leetcode/problem/2023-01-11-1443-minimum-time-to-collect-all-apples-in-a-tree/"
leetcode_ui: true
entry_slug: "2023-01-11-1443-minimum-time-to-collect-all-apples-in-a-tree"
---

[1443. Minimum Time to Collect All Apples in a Tree](https://leetcode.com/problems/minimum-time-to-collect-all-apples-in-a-tree/description/) medium

[https://t.me/leetcode_daily_unstoppable/82](https://t.me/leetcode_daily_unstoppable/82)

[blog post](https://leetcode.com/problems/minimum-time-to-collect-all-apples-in-a-tree/solutions/3036411/kotlin-build-tree-and-count-paths-to-parents/)

```kotlin
    fun minTime(n: Int, edges: Array<IntArray>, hasApple: List<Boolean>): Int {
        val graph = mutableMapOf<Int, MutableList<Int>>()
        edges.forEach { (from, to) ->
            graph.getOrPut(to, { mutableListOf() }) += from
            graph.getOrPut(from, { mutableListOf() }) += to
        }

        val queue = ArrayDeque<Int>()
        queue.add(0)
        val parents = IntArray(n+1) { it }
        while (queue.isNotEmpty()) {
            val node = queue.poll()
            graph[node]?.forEach {
                if (parents[it] == it && it != 0) {
                    parents[it] = node
                    queue.add(it)
                }
            }
        }
        var time = 0
        hasApple.forEachIndexed { i, has ->
            if (has) {
                var node = i
                while (node != parents[node]) {
                    val parent = parents[node]
                    parents[node] = node
                    node = parent
                    time++
                }
            }
        }
        return time * 2
    }

```

We need to count all paths from apples to 0-node and don't count already walked path.
* notice, that problem definition doesn't state the order of the edges in `edges` array. We need to build the tree first.

First, build the tree, let it be a `parents` array, where `parent[i]` is a parent of the `i`.
Walk graph with DFS and write the parents.
Next, walk `hasApple` list and for each apple count parents until reach node `0` or already visited node.
To mark a node as visited, make it the parent of itself.

Space: O(N), Time: O(N)

