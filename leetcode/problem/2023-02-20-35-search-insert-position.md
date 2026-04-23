---
layout: leetcode-entry
title: "35. Search Insert Position"
permalink: "/leetcode/problem/2023-02-20-35-search-insert-position/"
leetcode_ui: true
entry_slug: "2023-02-20-35-search-insert-position"
---

[35. Search Insert Position](https://leetcode.com/problems/search-insert-position/description/) easy

[blog post](https://leetcode.com/problems/search-insert-position/solutions/3208831/kotlin-binary-search/)

```kotlin

    fun searchInsert(nums: IntArray, target: Int): Int {
        var lo = 0
        var hi = nums.lastIndex
        while (lo <= hi) {
            val mid = lo + (hi - lo) / 2
            if (target == nums[mid]) return mid
            if (target > nums[mid]) lo = mid + 1
            else hi = mid - 1
        }
        return lo
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/124
#### Intuition
Just do a binary search

#### Approach
For more robust code consider:
* use only inclusive boundaries `lo` and `hi`
* loop also the last case when `lo == hi`
* always move boundaries `mid + 1` or `mid - 1`
* use distinct check for the exact match `nums[mid] == target`
* return `lo` position - this is an insertion point

#### Complexity
- Time complexity:
  $$O(log_2(n))$$
- Space complexity:
  $$O(1)$$

