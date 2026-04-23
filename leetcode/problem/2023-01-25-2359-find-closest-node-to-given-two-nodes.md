---
layout: leetcode-entry
title: "2359. Find Closest Node to Given Two Nodes"
permalink: "/leetcode/problem/2023-01-25-2359-find-closest-node-to-given-two-nodes/"
leetcode_ui: true
entry_slug: "2023-01-25-2359-find-closest-node-to-given-two-nodes"
---

[2359. Find Closest Node to Given Two Nodes](https://leetcode.com/problems/find-closest-node-to-given-two-nodes/description/) medium

[https://t.me/leetcode_daily_unstoppable/97](https://t.me/leetcode_daily_unstoppable/97)

[blog post](https://leetcode.com/problems/find-closest-node-to-given-two-nodes/solutions/3096815/kotlin-dfs/)

```kotlin
    fun closestMeetingNode(edges: IntArray, node1: Int, node2: Int): Int {
        val distances = mutableMapOf<Int, Int>()
        var n = node1
        var dist = 0
        while (n != -1) {
            if (distances.contains(n)) break
            distances[n] = dist
            n = edges[n]
            dist++
        }
        n = node2
        dist = 0
        var min = Int.MAX_VALUE
        var res = -1
        while (n != -1) {
            if (distances.contains(n)) {
                val one = distances[n]!!
                val max = maxOf(one, dist)
                if (max < min || max == min && n < res) {
                    min = max
                    res = n
                }
            }
            val tmp = edges[n]
            edges[n] = -1
            n = tmp
            dist++
        }
        return res
    }

```

![image.png](/assets/leetcode_daily_images/f97dcadd.webp)

We can walk with DFS and remember all distances, then compare them and choose those with minimum of maximums.
* we can use `visited` set, or modify an input
* corner case: don't forget to also store starting nodes

Space: O(n), Time: O(n)

