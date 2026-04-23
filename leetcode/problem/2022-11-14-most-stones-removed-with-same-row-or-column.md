---
layout: leetcode-entry
title: "Most Stones Removed With Same Row Or Column"
permalink: "/leetcode/problem/2022-11-14-most-stones-removed-with-same-row-or-column/"
leetcode_ui: true
entry_slug: "2022-11-14-most-stones-removed-with-same-row-or-column"
---

[https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/](https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/) medium

From observing the problem, we can see, that the task is in fact is to find an isolated islands:

```

        // * 3 *         * 3 *        * * *
        // 1 2 *    ->   * * *   or   1 * *
        // * * 4         * * 4        * * 4

        // * 3 *         * * *
        // 1 2 5    ->   * * *
        // * * 4         * * 4

```

```kotlin

    fun removeStones(stones: Array<IntArray>): Int {
        val uf = IntArray(stones.size) { it }
        var rootsCount = uf.size
        fun root(a: Int): Int {
            var x = a
            while (uf[x] != x) x = uf[x]
            return x
        }
        fun union(a: Int, b: Int) {
           val rootA = root(a)
           val rootB = root(b)
           if (rootA != rootB) {
               uf[rootA] = rootB
               rootsCount--
           }
        }
        val byY = mutableMapOf<Int, MutableList<Int>>()
        val byX = mutableMapOf<Int, MutableList<Int>>()
        stones.forEachIndexed { i, st ->
            byY.getOrPut(st[0], { mutableListOf() }).add(i)
            byX.getOrPut(st[1], { mutableListOf() }).add(i)
        }
        byY.values.forEach { list ->
            if (list.size > 1)
                for (i in 1..list.lastIndex) union(list[0], list[i])
        }
        byX.values.forEach { list ->
            if (list.size > 1)
                for (i in 1..list.lastIndex) union(list[0], list[i])
        }
        return stones.size - rootsCount
    }

```

Complexity: O(N)
Memory: O(N)

