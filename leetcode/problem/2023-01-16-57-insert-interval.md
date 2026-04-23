---
layout: leetcode-entry
title: "57. Insert Interval"
permalink: "/leetcode/problem/2023-01-16-57-insert-interval/"
leetcode_ui: true
entry_slug: "2023-01-16-57-insert-interval"
---

[57. Insert Interval](https://leetcode.com/problems/insert-interval/description/) medium

[https://t.me/leetcode_daily_unstoppable/87](https://t.me/leetcode_daily_unstoppable/87)

[blog post](https://leetcode.com/problems/insert-interval/solutions/3057540/kotlin-one-pass/)

```kotlin
    fun insert(intervals: Array<IntArray>, newInterval: IntArray): Array<IntArray> {
        val res = mutableListOf<IntArray>()
        var added = false
        fun add() {
            if (!added) {
                added = true
                if (res.isNotEmpty() && res.last()[1] >= newInterval[0]) {
                    res.last()[1] = maxOf(res.last()[1], newInterval[1])
                } else res += newInterval
            }
        }
        intervals.forEach { interval ->
            if (newInterval[0] <= interval[0]) add()

            if (res.isNotEmpty() && res.last()[1] >= interval[0]) {
                res.last()[1] = maxOf(res.last()[1], interval[1])
            } else  res += interval
        }
        add()

        return res.toTypedArray()
    }

```

There is no magic, just be careful with corner cases.

Make another list, and iterate interval, merging them and adding at the same time.
* don't forget to add `newInterval` if it is not added after iteration.

Space: O(N), Time: O(N)

