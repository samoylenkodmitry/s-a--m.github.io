---
layout: leetcode-entry
title: "2316. Count Unreachable Pairs of Nodes in an Undirected Graph"
permalink: "/leetcode/problem/2023-03-25-2316-count-unreachable-pairs-of-nodes-in-an-undirected-graph/"
leetcode_ui: true
entry_slug: "2023-03-25-2316-count-unreachable-pairs-of-nodes-in-an-undirected-graph"
---

[2316. Count Unreachable Pairs of Nodes in an Undirected Graph](https://leetcode.com/problems/count-unreachable-pairs-of-nodes-in-an-undirected-graph/description/) medium

[blog post](https://leetcode.com/problems/count-unreachable-pairs-of-nodes-in-an-undirected-graph/solutions/3338589/kotlin-union-find/)

```kotlin

fun countPairs(n: Int, edges: Array<IntArray>): Long {
    val uf = IntArray(n) { it }
    val sz = LongArray(n) { 1L }
    fun root(x: Int): Int {
        var n = x
        while (uf[n] != n) n = uf[n]
        uf[x] = n
        return n
    }
    fun union(a: Int, b: Int) {
        val rootA = root(a)
        val rootB = root(b)
        if (rootA != rootB) {
            uf[rootB] = rootA
            sz[rootA] += sz[rootB]
            sz[rootB] = 0L
        }
    }
    edges.forEach { (from, to) -> union(from, to) }
    // 1 2 4 = 1*2 + 1*4 + 2*4 = 1*2 + (1+2)*4
    var sum = 0L
    var count = 0L
    sz.forEach { // 2 2 4 = 2*2 + 2*4 + 2*4 = 2*2 + (2+2)*4
        count += sum * it
        sum += it
    }
    return count
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/159
#### Intuition
To find connected components sizes, we can use Union-Find.
To count how many pairs, we need to derive the formula, observing the pattern. Assume we have groups sizes `3, 4, 5`, the number of pairs is the number of pairs between `3,4` + the number of pairs between `4,5` + between `3,5`. Or, $$count(a,b,c) = count(a,b) + count(b,c) + count(a,c) $$ where $$count(a,b) = a*b$$. So, $$count_{abc} = ab + bc + ac = ab + (a + b)c = count_{ab} + (a+b)c$$, or $$count_i = count_{i-1} + x_i*\sum_{j=0}^{i}x$$
#### Approach
* use path compression for better `root` time complexity
#### Complexity
- Time complexity:
$$O(height)$$
- Space complexity:
$$O(n)$$

