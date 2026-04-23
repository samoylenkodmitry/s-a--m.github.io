---
layout: leetcode-entry
title: "1579. Remove Max Number of Edges to Keep Graph Fully Traversable"
permalink: "/leetcode/problem/2023-04-30-1579-remove-max-number-of-edges-to-keep-graph-fully-traversable/"
leetcode_ui: true
entry_slug: "2023-04-30-1579-remove-max-number-of-edges-to-keep-graph-fully-traversable"
---

[1579. Remove Max Number of Edges to Keep Graph Fully Traversable](https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/description/) hard

```kotlin

fun IntArray.root(a: Int): Int {
    var x = a
    while (this[x] != x) x = this[x]
    this[a] = x
    return x
}
fun IntArray.union(a: Int, b: Int): Boolean {
    val rootA = root(a)
    val rootB = root(b)
    if (rootA != rootB) this[rootB] = rootA
    return rootA != rootB
}
fun IntArray.connected(a: Int, b: Int) = root(a) == root(b)
fun maxNumEdgesToRemove(n: Int, edges: Array<IntArray>): Int {
    val uf1 = IntArray(n + 1) { it }
    val uf2 = IntArray(n + 1) { it }
    var skipped = 0
    edges.forEach { (type, a, b) ->
        if (type == 3) {
            uf1.union(a, b)
            if (!uf2.union(a, b)) skipped++
        }
    }
    edges.forEach { (type, a, b) ->
        if (type == 1 && !uf1.union(a, b)) skipped++
    }
    edges.forEach { (type, a, b) ->
        if (type == 2 && !uf2.union(a, b)) skipped++
    }
    for (i in 2..n)
    if (!uf1.connected(i - 1, i) || !uf2.connected(i - 1, i)) return -1
    return skipped
}

```

[blog post](https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/solutions/3468491/kotlin-union-find/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-30042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/196
#### Intuition
After connecting all `type 3` nodes, we can skip already connected nodes for Alice and for Bob. To detect if all the nodes are connected, we can just check if all nodes connected to one particular node.
#### Approach
Use separate `Union-Find` objects for Alice and for Bob
#### Complexity
- Time complexity:
$$O(n)$$, as `root` and `union` operations take `< 5` for any `n <= Int.MAX`.
- Space complexity:
$$O(n)$$

