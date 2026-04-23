---
layout: leetcode-entry
title: "1971. Find if Path Exists in Graph"
permalink: "/leetcode/problem/2022-12-19-1971-find-if-path-exists-in-graph/"
leetcode_ui: true
entry_slug: "2022-12-19-1971-find-if-path-exists-in-graph"
---

[1971. Find if Path Exists in Graph](https://leetcode.com/problems/find-if-path-exists-in-graph/description/) easy

[https://t.me/leetcode_daily_unstoppable/57](https://t.me/leetcode_daily_unstoppable/57)

[blog post](https://leetcode.com/problems/find-if-path-exists-in-graph/solutions/2928882/kotlin-bfs/)

```kotlin
    fun validPath(n: Int, edges: Array<IntArray>, source: Int, destination: Int): Boolean {
        if (source == destination) return true
        val graph = mutableMapOf<Int, MutableList<Int>>()
        edges.forEach { (from, to) ->
            graph.getOrPut(from, { mutableListOf() }).add(to)
            graph.getOrPut(to, { mutableListOf() }).add(from)
        }
        val visited = mutableSetOf<Int>()
        with(ArrayDeque<Int>()) {
            add(source)
            var depth = 0
            while(isNotEmpty() && ++depth < n) {
                repeat(size) {
                    graph[poll()]?.forEach {
                        if (it == destination) return true
                        if (visited.add(it)) add(it)
                    }
                }
            }
        }
        return false
    }

```

BFS will do the job.
Make node to nodes map, keep visited set and use queue for BFS.
* also path can't be longer than n elements

Space: O(N), Time: O(N)

