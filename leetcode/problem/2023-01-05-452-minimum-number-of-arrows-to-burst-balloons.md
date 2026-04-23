---
layout: leetcode-entry
title: "452. Minimum Number of Arrows to Burst Balloons"
permalink: "/leetcode/problem/2023-01-05-452-minimum-number-of-arrows-to-burst-balloons/"
leetcode_ui: true
entry_slug: "2023-01-05-452-minimum-number-of-arrows-to-burst-balloons"
---

[452. Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/description/) medium

[https://t.me/leetcode_daily_unstoppable/75](https://t.me/leetcode_daily_unstoppable/75)

[blog post](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/solutions/3002258/kotlin-sort-line-sweep/)

```kotlin
    fun findMinArrowShots(points: Array<IntArray>): Int {
        if (points.isEmpty()) return 0
        if (points.size == 1) return 1
        Arrays.sort(points, Comparator<IntArray> { a, b ->
            if (a[0] == b[0]) a[1].compareTo(b[1]) else a[0].compareTo(b[0]) })
        var arrows = 1
        var arrX = points[0][0]
        var minEnd = points[0][1]
        for (i in 1..points.lastIndex) {
            val (start, end) = points[i]
            if (minEnd < start) {
                arrows++
                minEnd = end
            }
            if (end < minEnd) minEnd = end
            arrX = start
        }
        return arrows
    }

```

The optimal strategy to achieve the minimum number of arrows is to find the maximum overlapping intervals. For this task, we can sort the points by their `start` and `end` coordinates and use line sweep technique. Overlapping intervals are separate if their `minEnd` is less than `start` of the next interval. `minEnd` - the minimum of the `end`'s of the overlapping intervals.
Let's move the arrow to each `start` interval and fire a new arrow if this `start` is greater than `minEnd`.
* for sorting without Int overflowing, use `compareTo` instead of subtraction
* initial conditions are better to initialize with the first interval and iterate starting from the second

Space: O(1), Time: O(NlogN)

