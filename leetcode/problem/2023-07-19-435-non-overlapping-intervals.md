---
layout: leetcode-entry
title: "435. Non-overlapping Intervals"
permalink: "/leetcode/problem/2023-07-19-435-non-overlapping-intervals/"
leetcode_ui: true
entry_slug: "2023-07-19-435-non-overlapping-intervals"
---

[435. Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/description/) medium
[blog post](https://leetcode.com/problems/non-overlapping-intervals/solutions/3785669/kotlin-line-sweep/)
[substack](https://dmitriisamoilenko.substack.com/p/19072023-435-non-overlapping-intervals?sd=pf)
![image.png](/assets/leetcode_daily_images/0816cd0c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/280

#### Problem TLDR

Minimum intervals to erase overlap

#### Intuition

First idea, is to sort the array by `from`. Next, we can greedily take intervals and remove overlapping ones. But, to remove the `minimum` number, we can start with removing the most `long` intervals.

#### Approach

* walk the sweep line, counting how many intervals are non overlapping
* only move the `right border` when there is a new non overlapping interval
* minimize the `border` when it shrinks

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun eraseOverlapIntervals(intervals: Array<IntArray>): Int {
        intervals.sortWith(compareBy({ it[0] }))
        var border = Int.MIN_VALUE
        return intervals.count { (from, to) ->
          (border > from).also {
            if (border <= from || border > to) border = to
          }
        }
    }

```

