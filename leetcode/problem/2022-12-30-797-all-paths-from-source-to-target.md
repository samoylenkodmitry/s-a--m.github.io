---
layout: leetcode-entry
title: "797. All Paths From Source to Target"
permalink: "/leetcode/problem/2022-12-30-797-all-paths-from-source-to-target/"
leetcode_ui: true
entry_slug: "2022-12-30-797-all-paths-from-source-to-target"
---

[797. All Paths From Source to Target](https://leetcode.com/problems/all-paths-from-source-to-target/description/) medium

[https://t.me/leetcode_daily_unstoppable/68](https://t.me/leetcode_daily_unstoppable/68)

[blog post](https://leetcode.com/problems/all-paths-from-source-to-target/solutions/1600383/kotlin-dfs-backtracking-java-iterative-dfs-stack/)

```kotlin
    fun allPathsSourceTarget(graph: Array<IntArray>): List<List<Int>> {
        val res = mutableListOf<List<Int>>()
        val currPath = mutableListOf<Int>()
        fun dfs(curr: Int) {
            currPath += curr
            if (curr == graph.lastIndex) res += currPath.toList()
            graph[curr].forEach { dfs(it) }
            currPath.removeAt(currPath.lastIndex)
        }
        dfs(0)
        return res
    }

```

We must find all the paths, so there is no shortcuts to the visiting all of them.
One technique is backtracking - reuse existing visited list of nodes.

Space: O(VE), Time: O(VE)

