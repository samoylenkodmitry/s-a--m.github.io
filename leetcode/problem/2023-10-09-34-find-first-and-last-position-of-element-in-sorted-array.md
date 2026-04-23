---
layout: leetcode-entry
title: "34. Find First and Last Position of Element in Sorted Array"
permalink: "/leetcode/problem/2023-10-09-34-find-first-and-last-position-of-element-in-sorted-array/"
leetcode_ui: true
entry_slug: "2023-10-09-34-find-first-and-last-position-of-element-in-sorted-array"
---

[34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/) medium
[blog post](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/solutions/4148104/kotlin-bs/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/9102023-34-find-first-and-last-position?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/970b7ba3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/364

#### Problem TLDR

Binary Search range

#### Intuition

Just write a Binary Search

#### Approach

For simpler code:
* use inclusive `lo` and `hi`
* check the last condition `lo == hi`
* always move the borders `lo = mid + 1`, `hi = mid - 1`
* always write the found result `if (nums[mid] == target)`
* to understand which border to move, consider this thought: `if this position is definitely less than target, we can drop it and all that less than it`

#### Complexity

- Time complexity:
$$O(log(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun searchRange(nums: IntArray, target: Int): IntArray {
      var from = -1
      var lo = 0
      var hi = nums.lastIndex
      while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        if (nums[mid] == target) from = min(max(from, nums.size), mid)
        if (nums[mid] < target) lo = mid + 1
        else hi = mid - 1
      }
      var to = from
      lo = maxOf(0, from)
      hi = nums.lastIndex
      while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        if (nums[mid] == target) to = max(min(-1, to), mid)
        if (nums[mid] <= target) lo = mid + 1
        else hi = mid - 1
      }
      return intArrayOf(from, to)
    }

```

