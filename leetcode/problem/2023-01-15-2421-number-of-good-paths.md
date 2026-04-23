---
layout: leetcode-entry
title: "2421. Number of Good Paths"
permalink: "/leetcode/problem/2023-01-15-2421-number-of-good-paths/"
leetcode_ui: true
entry_slug: "2023-01-15-2421-number-of-good-paths"
---

[2421. Number of Good Paths](https://leetcode.com/problems/number-of-good-paths/) hard

[https://t.me/leetcode_daily_unstoppable/86](https://t.me/leetcode_daily_unstoppable/86)

[blog post](https://leetcode.com/problems/number-of-good-paths/solutions/3054534/kotlin-union-find-was-hard/)

```kotlin
    fun numberOfGoodPaths(vals: IntArray, edges: Array<IntArray>): Int {
        if (edges.size == 0) return vals.size
        edges.sortWith(compareBy(  { maxOf( vals[it[0]], vals[it[1]] ) }  ))
        val uf = IntArray(vals.size) { it }
        val freq = Array(vals.size) { mutableMapOf(vals[it] to 1) }
        fun find(x: Int): Int {
            var p = x
            while (uf[p] != p) p = uf[p]
            uf[x] = p
            return p
        }
        fun union(a: Int, b: Int): Int {
            val rootA = find(a)
            val rootB = find(b)
            if (rootA == rootB) return 0
            uf[rootA] = rootB
            val vMax = maxOf(vals[a], vals[b]) // if we connect tree [1-3] to tree [2-1], only `3` matters
            val countA = freq[rootA][vMax] ?:0
            val countB = freq[rootB][vMax] ?:0
            freq[rootB][vMax] = countA + countB
            return countA * countB
        }
        return edges.map { union(it[0], it[1])}.sum()!! + vals.size
    }

```

The naive solution with single DFS and merging frequency maps gives TLE.
Now, use hint, and they tell you to sort edges and use Union-Find :)
The idea is to connect subtrees, but walk them from smallest to the largest of value.
When we connect two subtrees, we look at the maximum of each subtree.
The minimum values don't matter because the path will break at the maximums by definition of the problem.

Use IntArray for Union-Find, and also keep frequencies maps for each root.

Space: O(NlogN), Time: O(N)

