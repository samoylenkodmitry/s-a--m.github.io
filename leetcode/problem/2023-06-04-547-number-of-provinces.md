---
layout: leetcode-entry
title: "547. Number of Provinces"
permalink: "/leetcode/problem/2023-06-04-547-number-of-provinces/"
leetcode_ui: true
entry_slug: "2023-06-04-547-number-of-provinces"
---

[547. Number of Provinces](https://leetcode.com/problems/number-of-provinces/description/) medium
[blog post](https://leetcode.com/problems/number-of-provinces/solutions/3594857/kotlin-union-find/)
[substack](https://dmitriisamoilenko.substack.com/p/04062023-547-number-of-provinces?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/235
#### Problem TLDR
Count connected groups in graph.
#### Intuition
Union-Find will perfectly fit to solve this problem.

#### Approach
For more optimal Union-Find:
* use path compression in the `root` method: `uf[it] = x`
* connect the smallest size subtree to the largest
#### Complexity
- Time complexity:
$$O(a(n)n^2)$$, `a(n)` - reverse Ackerman function `f(x) = 2^2^2..^2, x times`. `a(Int.MAX_VALUE) = 2^32 = 2^2^5 == 3`
- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

fun findCircleNum(isConnected: Array<IntArray>): Int {
    val uf = IntArray(isConnected.size) { it }
    val sz = IntArray(isConnected.size) { 1 }
    var count = uf.size
    val root: (Int) -> Int = {
        var x = it
        while (uf[x] != x) x = uf[x]
        uf[it] = x
        x
    }
    val connect: (Int, Int) -> Unit = { a, b ->
        val rootA = root(a)
        val rootB = root(b)
        if (rootA != rootB) {
            count--
            if (sz[rootA] < sz[rootB]) {
                uf[rootB] = rootA
                sz[rootA] += sz[rootB]
                sz[rootB] = 0
            } else {
                uf[rootA] = rootB
                sz[rootB] += sz[rootA]
                sz[rootA] = 0
            }
        }
    }
    for (i in 0..sz.lastIndex)
    for (j in 0..sz.lastIndex)
    if (isConnected[i][j] == 1) connect(i, j)
    return count
}

```

