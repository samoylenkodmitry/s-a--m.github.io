---
layout: leetcode-entry
title: "1615. Maximal Network Rank"
permalink: "/leetcode/problem/2023-08-18-1615-maximal-network-rank/"
leetcode_ui: true
entry_slug: "2023-08-18-1615-maximal-network-rank"
---

[1615. Maximal Network Rank](https://leetcode.com/problems/maximal-network-rank/description/) medium
[blog post](https://leetcode.com/problems/maximal-network-rank/solutions/3924953/kotlin-n-2/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18082023-1615-maximal-network-rank?utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/418638b6.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/312

#### Problem TLDR

Max edges count for each pair of nodes

#### Intuition

We can just count edges for each node, then search for max in an n^2 for-loop.

#### Approach

* use a `HashSet` to check `contains` in O(1)

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$, there are up to n^2 edges

#### Code

```kotlin

    fun maximalNetworkRank(n: Int, roads: Array<IntArray>): Int {
        val fromTo = mutableMapOf<Int, HashSet<Int>>()
        roads.forEach { (from, to) ->
          fromTo.getOrPut(from) { HashSet() } += to
          fromTo.getOrPut(to) { HashSet() } += from
        }
        var max = 0
        for (a in 0 until n) {
          for (b in a + 1 until n) {
            val countA = fromTo[a]?.size ?: 0
            val countB = fromTo[b]?.size ?: 0
            val direct = fromTo[a]?.contains(b) ?: false
            max = maxOf(max, countA + countB - (if (direct) 1 else 0))
          }
        }
        return max
    }

```

