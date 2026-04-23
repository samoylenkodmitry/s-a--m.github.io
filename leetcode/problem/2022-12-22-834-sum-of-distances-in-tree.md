---
layout: leetcode-entry
title: "834. Sum of Distances in Tree"
permalink: "/leetcode/problem/2022-12-22-834-sum-of-distances-in-tree/"
leetcode_ui: true
entry_slug: "2022-12-22-834-sum-of-distances-in-tree"
---

[834. Sum of Distances in Tree](https://leetcode.com/problems/sum-of-distances-in-tree/description/) hard

[https://t.me/leetcode_daily_unstoppable/60](https://t.me/leetcode_daily_unstoppable/60)

[blog post](https://leetcode.com/problems/sum-of-distances-in-tree/solutions/1443979/kotlin-java-2-dfs-diagramm-to-invent-the-change-root-equation/)

```kotlin
    fun sumOfDistancesInTree(n: Int, edges: Array<IntArray>): IntArray {
        val graph = mutableMapOf<Int, MutableList<Int>>()
        edges.forEach { (from, to) ->
            graph.getOrPut(from, { mutableListOf() }) += to
            graph.getOrPut(to, { mutableListOf() }) += from
        }
        val counts = IntArray(n) { 1 }
        val sums = IntArray(n) { 0 }
        fun distSum(pos: Int, visited: Int) {
            graph[pos]?.forEach {
                if (it != visited) {
                    distSum(it, pos)
                    counts[pos] += counts[it]
                    sums[pos] += counts[it] + sums[it]
                }
            }
        }
        fun dfs(pos: Int, visited: Int) {
            graph[pos]?.forEach {
                if (it != visited) {
                    sums[it] = sums[pos] - counts[it] + (n - counts[it])
                    dfs(it, pos)
                }
            }
        }
        distSum(0, -1)
        dfs(0, -1)
        return sums
    }

```

We can do the job for item #0, then we need to invent a formula to reuse some data when we change the node.

How to mathematically prove formula for a new sum:
![image](/assets/leetcode_daily_images/202464c8.webp)

![image.png](/assets/leetcode_daily_images/89f3217e.webp)
Store count of children in a `counts` array, and sum of the distances to children in a `dist` array. In a first DFS traverse from a node 0 and fill the arrays. In a second DFS only modify `dist` based on previous computed `dist` value, using formula: `sum[curr] = sum[prev] - count[curr] + (N - count[curr])`

Space: O(N), Time: O(N)

