---
layout: leetcode-entry
title: "1319. Number of Operations to Make Network Connected"
permalink: "/leetcode/problem/2023-03-23-1319-number-of-operations-to-make-network-connected/"
leetcode_ui: true
entry_slug: "2023-03-23-1319-number-of-operations-to-make-network-connected"
---

[1319. Number of Operations to Make Network Connected](https://leetcode.com/problems/number-of-operations-to-make-network-connected/description/) medium

[blog post](https://leetcode.com/problems/number-of-operations-to-make-network-connected/solutions/3331235/kotlin-union-find/)

```kotlin

fun makeConnected(n: Int, connections: Array<IntArray>): Int {
    var extraCables = 0
    var groupsCount = n
    val uf = IntArray(n) { it }
    fun findRoot(x: Int): Int {
        var n = x
        while (uf[n] != n) n = uf[n]
        uf[x] = n
        return n
    }
    fun connect(a: Int, b: Int) {
        val rootA = findRoot(a)
        val rootB = findRoot(b)
        if (rootA == rootB) {
            extraCables++
            return
        }
        uf[rootB] = rootA
        groupsCount--
    }
    connections.forEach { (from, to) -> connect(from, to) }
    return if (extraCables < groupsCount - 1) -1 else groupsCount - 1
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/157
#### Intuition
The number of cables we need is the number of disconnected groups of connected computers. Cables can be taken from the computers that have extra connections. We can do this using BFS/DFS and tracking visited set, counting extra cables if already visited node is in connection.
Another solution is to use Union-Find for the same purpose.

#### Approach
* for the better time complexity of the `findRoot` use path compression: `uf[x] = n`
#### Complexity
- Time complexity:
$$O(n*h)$$, $$h$$ - tree height, in a better implementation, can be down to constant. For Quick-Union-Find it is lg(n).
- Space complexity:
$$O(n)$$

