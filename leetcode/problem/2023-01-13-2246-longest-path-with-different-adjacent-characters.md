---
layout: leetcode-entry
title: "2246. Longest Path With Different Adjacent Characters"
permalink: "/leetcode/problem/2023-01-13-2246-longest-path-with-different-adjacent-characters/"
leetcode_ui: true
entry_slug: "2023-01-13-2246-longest-path-with-different-adjacent-characters"
---

[2246. Longest Path With Different Adjacent Characters](https://leetcode.com/problems/longest-path-with-different-adjacent-characters/description/) hard

[https://t.me/leetcode_daily_unstoppable/84](https://t.me/leetcode_daily_unstoppable/84)

[blog post](https://leetcode.com/problems/longest-path-with-different-adjacent-characters/solutions/3046179/kotlin-build-graph-dfs/)

```kotlin
    fun longestPath(parent: IntArray, s: String): Int {
        val graph = mutableMapOf<Int, MutableList<Int>>()
        for (i in 1..parent.lastIndex)
            if (s[i] != s[parent[i]]) graph.getOrPut(parent[i], { mutableListOf() }) += i

        var maxLen = 0
        fun dfs(curr: Int): Int {
            parent[curr] = curr
            var max1 = 0
            var max2 = 0
            graph[curr]?.forEach {
                val childLen = dfs(it)
                if (childLen > max1) {
                    max2 = max1
                    max1 = childLen
                } else if (childLen > max2) max2 = childLen
            }
            val childChainLen = 1 + (max1 + max2)
            val childMax = 1 + max1
            maxLen = maxOf(maxLen, childMax, childChainLen)
            return childMax
        }
        for (i in 0..parent.lastIndex) if (parent[i] != i) dfs(i)

        return maxLen
    }

```

Longest path is a maximum sum of the two longest paths of the current node.

Let's build a graph and then recursively iterate it by DFS. We need to find two largest results from the children DFS calls.
* make `parent[i] == i` to store a `visited` state

Space: O(N), Time: O(N), in DFS we visit each node only once.

