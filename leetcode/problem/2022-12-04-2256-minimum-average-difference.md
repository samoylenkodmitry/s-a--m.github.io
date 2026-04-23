---
layout: leetcode-entry
title: "2256. Minimum Average Difference"
permalink: "/leetcode/problem/2022-12-04-2256-minimum-average-difference/"
leetcode_ui: true
entry_slug: "2022-12-04-2256-minimum-average-difference"
---

[2256. Minimum Average Difference](https://leetcode.com/problems/minimum-average-difference/) medium

[https://t.me/leetcode_daily_unstoppable/41](https://t.me/leetcode_daily_unstoppable/41)

```kotlin

    fun minimumAverageDifference(nums: IntArray): Int {
        var sum = 0L
        nums.forEach { sum += it.toLong() }
        var leftSum = 0L
        var min = Long.MAX_VALUE
        var minInd = 0
        for (i in 0..nums.lastIndex) {
            val leftCount = (i+1).toLong()
            leftSum += nums[i].toLong()
            val front = leftSum/leftCount
            val rightCount = nums.size.toLong() - leftCount
            val rightSum = sum - leftSum
            val back = if (rightCount == 0L) 0L else rightSum/rightCount
            val diff = Math.abs(front - back)
            if (diff < min) {
                min = diff
                minInd = i
            }
        }
        return minInd
    }

```

### Intuition

Two pointers, one for even, one for odd indexes.
### Approach

To avoid mistakes you need to be verbose, and don't skip operations:
* store evenHead in a separate variable
* don't switch links before both pointers jumped
* don't make odd pointer null
* try to run for simple input `1->2->null` by yourself

Space: O(1), Time: O(n)

