---
layout: leetcode-entry
title: "540. Single Element in a Sorted Array"
permalink: "/leetcode/problem/2023-02-21-540-single-element-in-a-sorted-array/"
leetcode_ui: true
entry_slug: "2023-02-21-540-single-element-in-a-sorted-array"
---

[540. Single Element in a Sorted Array](https://leetcode.com/problems/single-element-in-a-sorted-array/description/) medium

[blog post](https://leetcode.com/problems/single-element-in-a-sorted-array/solutions/3213551/kotlin-odd-even-positions-binary-search/)

```kotlin

fun singleNonDuplicate(nums: IntArray): Int {
    var lo = 0
    var hi = nums.lastIndex
    // 0 1 2 3 4
    // 1 1 2 3 3
    while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        val prev = if (mid > 0) nums[mid-1] else -1
        val next = if (mid < nums.lastIndex) nums[mid+1] else Int.MAX_VALUE
        val curr = nums[mid]
        if (prev < curr && curr < next) return curr

        val oddPos = mid % 2 != 0
        val isSingleOnTheLeft = oddPos && curr == next || !oddPos && curr == prev

        if (isSingleOnTheLeft) hi = mid - 1 else lo = mid + 1
    }
    return -1
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/125
#### Intuition
This problem is a brain-teaser until you notice that pairs are placed at `even-odd` positions before the target and at `odd-even` positions after.
#### Approach
Let's write a binary search. For more robust code, consider:
* use inclusive `lo` and `hi`
* always move `lo` or `hi`
* check for the target condition and return early
#### Complexity
- Time complexity:
$$O(log_2(n))$$
- Space complexity:
$$O(1)$$

