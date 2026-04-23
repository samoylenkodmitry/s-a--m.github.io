---
layout: leetcode-entry
title: "918. Maximum Sum Circular Subarray"
permalink: "/leetcode/problem/2023-01-18-918-maximum-sum-circular-subarray/"
leetcode_ui: true
entry_slug: "2023-01-18-918-maximum-sum-circular-subarray"
---

[918. Maximum Sum Circular Subarray](https://leetcode.com/problems/maximum-sum-circular-subarray/description/) medium

[https://t.me/leetcode_daily_unstoppable/89](https://t.me/leetcode_daily_unstoppable/89)

[blog post](https://leetcode.com/problems/maximum-sum-circular-subarray/solutions/3069120/kotlin-invert-the-problem/)

```kotlin
    fun maxSubarraySumCircular(nums: IntArray): Int {
        var maxEndingHere = 0
        var maxEndingHereNegative = 0
        var maxSoFar = Int.MIN_VALUE
        var total = nums.sum()
        nums.forEach {
            maxEndingHere += it
            maxEndingHereNegative += -it
            maxSoFar = maxOf(maxSoFar, maxEndingHere, if (total == -maxEndingHereNegative) Int.MIN_VALUE else total+maxEndingHereNegative)
            if (maxEndingHere < 0) {
                maxEndingHere = 0
            }
            if (maxEndingHereNegative < 0) {
                maxEndingHereNegative = 0
            }
        }
        return maxSoFar
    }

```

Simple Kadane's Algorithm didn't work when we need to keep a window of particular size.
One idea is to invert the problem and find the minimum sum and subtract it from the total.

One corner case:
* we can't subtract all the elements when checking the negative sum.

Space: O(1), Time: O(N)

