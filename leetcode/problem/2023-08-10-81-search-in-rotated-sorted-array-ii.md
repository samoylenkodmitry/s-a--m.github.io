---
layout: leetcode-entry
title: "81. Search in Rotated Sorted Array II"
permalink: "/leetcode/problem/2023-08-10-81-search-in-rotated-sorted-array-ii/"
leetcode_ui: true
entry_slug: "2023-08-10-81-search-in-rotated-sorted-array-ii"
---

[81. Search in Rotated Sorted Array II](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/description/) medium
[blog post](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/solutions/3888620/kotlin-binary-seach/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10082023-81-search-in-rotated-sorted?utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/be533803.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/304

#### Problem TLDR

Binary Search in a rotated array with duplicates

#### Intuition

There are several cases:
* pivot on the left, right side can be checked
* pivot on the right, left side can be checked
* nums[lo] == nums[hi], do a linear scan

#### Approach

For more robust code:
* inclusive `lo` and `hi`
* last check `lo == hi`
* check the result `nums[mid] == target`
* move borders `lo = mid + 1`, `hi = mid - 1`
* exclusive checks `<` & `>` are simpler to reason about than inclusive `<=`, `=>`

#### Complexity

- Time complexity:
$$O(n)$$, the worst case is linear in a long array of duplicates

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun search(nums: IntArray, target: Int): Boolean {
        var lo = 0
        var hi = nums.lastIndex
        while (lo <= hi) {
          val mid = lo + (hi - lo) / 2
          if (nums[mid] == target) return true
          if (nums[lo] < nums[hi]) { // normal case
            if (nums[mid] < target) lo = mid + 1 else hi = mid - 1
          } else if (nums[lo] > nums[hi]) { // pivot case
            if (nums[mid] > nums[hi]) {
              // pivot on the right
              // 5 6 7 8 9 1 2
              //   t   m   p
              if (target in nums[lo]..nums[mid]) hi = mid - 1 else lo = mid + 1
            } else {
              // pivot on the left
              //   9 1 2 3 4
              //     p m t
              if (target in nums[mid]..nums[hi]) lo = mid + 1 else hi = mid - 1
            }
          } else hi-- // nums[lo] == nums[hi]
        }
        return false
    }

```

