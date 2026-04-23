---
layout: leetcode-entry
title: "2492. Minimum Score of a Path Between Two Cities"
permalink: "/leetcode/problem/2023-03-22-2492-minimum-score-of-a-path-between-two-cities/"
leetcode_ui: true
entry_slug: "2023-03-22-2492-minimum-score-of-a-path-between-two-cities"
---

[2492. Minimum Score of a Path Between Two Cities](https://leetcode.com/problems/minimum-score-of-a-path-between-two-cities/description/) medium

[blog post](https://leetcode.com/problems/minimum-score-of-a-path-between-two-cities/solutions/3327604/kotlin-union-find/)

```kotlin

fun minScore(n: Int, roads: Array<IntArray>): Int {
    val uf = Array(n + 1) { it }
    val minDist = Array(n + 1) { Int.MAX_VALUE }
    fun findRoot(x: Int): Int {
        var n = x
        while (uf[n] != n) n = uf[n]
        uf[x] = n
        return n
    }
    fun union(a: Int, b: Int, dist: Int) {
        val rootA = findRoot(a)
        val rootB = findRoot(b)
        uf[rootB] = rootA
        minDist[rootA] = minOf(minDist[rootA], minDist[rootB], dist)
    }
    roads.forEach { (from, to, dist) -> union(from, to, dist) }
    return minDist[findRoot(1)]
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/156
#### Intuition
Observing the problem definition, we don't care about the path, but only about the minimum distance in a connected subset containing `1` and `n`. This can be solved by simple BFS, which takes $$O(V+E)$$ time and space. But ideal data structure for this problem is Union-Find.
* In an interview, it is better to just start with BFS, because explaining the time complexity of the `find` operation of Union-Find is difficult. https://algs4.cs.princeton.edu/15uf/

#### Approach
Connect all roads and update minimums in the Union-Find data structure. Use simple arrays for both connections and minimums.
* updating a root after finding it gives more optimal time
#### Complexity
- Time complexity:
$$O(E*tree_height)$$
- Space complexity:
$$O(n)$$

