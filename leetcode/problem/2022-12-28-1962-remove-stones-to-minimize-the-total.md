---
layout: leetcode-entry
title: "1962. Remove Stones to Minimize the Total"
permalink: "/leetcode/problem/2022-12-28-1962-remove-stones-to-minimize-the-total/"
leetcode_ui: true
entry_slug: "2022-12-28-1962-remove-stones-to-minimize-the-total"
---

[1962. Remove Stones to Minimize the Total](https://leetcode.com/problems/remove-stones-to-minimize-the-total/description/) medium

[https://t.me/leetcode_daily_unstoppable/66](https://t.me/leetcode_daily_unstoppable/66)

[blog post](https://leetcode.com/problems/remove-stones-to-minimize-the-total/solutions/2961725/kotlin-priorityqueue/)

```kotlin
    fun minStoneSum(piles: IntArray, k: Int): Int {
        val pq = PriorityQueue<Int>()
        var sum = 0
        piles.forEach {
            sum += it
            pq.add(-it)
        }
        for (i in 1..k) {
            if (pq.isEmpty()) break
            val max = -pq.poll()
            if (max == 0) break
            val newVal = Math.round(max/2.0).toInt()
            sum -= max - newVal
            pq.add(-newVal)
        }
        return sum
    }

```

By the problem definition, intuitively the best strategy is to reduce the maximum each time.
Use `PriorityQueue` to keep track of the maximum value and update it dynamically.
* one can use variable `sum` and update it each time.

Space: O(n), Time: O(nlogn)

