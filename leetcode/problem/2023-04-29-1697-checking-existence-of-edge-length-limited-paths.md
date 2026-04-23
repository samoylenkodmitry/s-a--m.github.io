---
layout: leetcode-entry
title: "1697. Checking Existence of Edge Length Limited Paths"
permalink: "/leetcode/problem/2023-04-29-1697-checking-existence-of-edge-length-limited-paths/"
leetcode_ui: true
entry_slug: "2023-04-29-1697-checking-existence-of-edge-length-limited-paths"
---

[1697. Checking Existence of Edge Length Limited Paths](https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/description/) hard

```kotlin

fun distanceLimitedPathsExist(n: Int, edgeList: Array<IntArray>, queries: Array<IntArray>): BooleanArray {
    val uf = IntArray(n) { it }
    fun root(x: Int): Int {
        var n = x
        while (uf[n] != n) n = uf[n]
        uf[x] = n
        return n
    }
    fun union(a: Int, b: Int) {
        val rootA = root(a)
        val rootB = root(b)
        if (rootA != rootB) uf[rootB] = rootA
    }
    val indices = queries.indices.sortedWith(compareBy( { queries[it][2] } ))
    edgeList.sortWith(compareBy( { it[2] } ))
    var edgePos = 0
    val res = BooleanArray(queries.size)
    indices.forEach { ind ->
        val (qfrom, qto, maxDist) = queries[ind]
        while (edgePos < edgeList.size) {
            val (from, to, dist) = edgeList[edgePos]
            if (dist >= maxDist) break
            union(from, to)
            edgePos++
        }
        res[ind] = root(qfrom) == root(qto)
    }
    return res
}

```

[blog post](https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/solutions/3465266/kotlin-union-islands/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-29042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/195
#### Intuition
The naive approach is to do BFS for each query, obviously gives TLE as it takes $$O(n^2)$$ time.
Using the hint, we can use somehow the sorted order of the queries. If we connect every two nodes with `dist < query.dist` we have connected groups with all nodes reachable inside them. The best data structure for union and finding connected groups is the Union-Find.
To avoid iterating `edgeList` every time, we can sort it too and take only available distances.

#### Approach
* for better time complexity, compress the Union-Find path `uf[x] = n`
* track the `edgePos` - a position in a sorted `edgeList`
* make separate `indices` list to sort queries without losing the order
#### Complexity
- Time complexity:
$$O(nlog(n))$$, time complexity for `root` and `union` operations is an inverse Ackerman function and `< 5` for every possible number in Int.
- Space complexity:
$$O(n)$$

