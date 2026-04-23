---
layout: leetcode-entry
title: "1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree"
permalink: "/leetcode/problem/2023-08-19-1489-find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/"
leetcode_ui: true
entry_slug: "2023-08-19-1489-find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree"
---

[1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree](https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/description/) hard
[blog post](https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/solutions/3929582/kotlin-union-find/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19082023-1489-find-critical-and-pseudo?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/9aa73d6b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/313

#### Problem TLDR

List of list of `must-have` edges and list of `optional` edges for Minimum Weight Minimum Spanning Tree

#### Intuition

Use hints.

Minimum Spanning Tree can be obtained by sorting edges and adding not connected one-by-one using Union-Find

After we found `target` minimum weight, we can check how each node contributes: if removing the node increases the `target`, the node is a `must-have`. Also, if force using node in a spanning tree doesn't change the `target`, node is `optional`.

#### Approach

* careful with the sorted order of indices, returned positions must be in initial order
* check if spanning tree is impossible to make, by checking if all nodes are connected

#### Complexity

- Time complexity:
$$O(E^2 + EV)$$, sorting edges takes `ElogE`, then cycle `E` times algorithm of `E+V`

- Space complexity:
$$O(E + V)$$, `E` for sorted edges, `V` for Union-Find array

#### Code

```kotlin

    fun findCriticalAndPseudoCriticalEdges(n: Int, edges: Array<IntArray>): List<List<Int>> {
      val sorted = edges.indices.sortedWith(compareBy({ edges[it][2] }))
      fun minSpanTreeW(included: Int = -1, excluded: Int = -1): Int {
        val uf = IntArray(n) { it }
        fun find(x: Int): Int = if (x == uf[x]) x else find(uf[x]).also { uf[x] = it }
        fun union(ind: Int): Int {
          val (a, b, w) = edges[ind]
          return if (find(a) == find(b)) 0 else w.also { uf[find(b)] = find(a) }
        }
        return ((if (included < 0) 0 else union(included)) + sorted
          .filter { it != excluded }.map { union(it) }.sum()!!)
          .takeIf { (0 until n).all { find(0) == find(it) } } ?: Int.MAX_VALUE
      }
      val target = minSpanTreeW()
      val critical = mutableListOf<Int>()
      val pseudo = mutableListOf<Int>()
      edges.indices.forEach {
        if (minSpanTreeW(excluded = it)  > target) critical += it
        else if (minSpanTreeW(included = it) == target) pseudo += it
      }
      return listOf(critical, pseudo)
    }

```

