---
layout: leetcode-entry
title: "33. Search in Rotated Sorted Array"
permalink: "/leetcode/problem/2023-08-08-33-search-in-rotated-sorted-array/"
leetcode_ui: true
entry_slug: "2023-08-08-33-search-in-rotated-sorted-array"
---

[33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/description/) medium
[blog post](https://leetcode.com/problems/search-in-rotated-sorted-array/solutions/3879746/kotlin-binary-search/)
[substack](https://dmitriisamoilenko.substack.com/p/08082023-33-search-in-rotated-sorted?sd=pf)
![image.png](/assets/leetcode_daily_images/30dc6684.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/302

#### Problem TLDR

Binary Search in a shifted array

#### Intuition

The special case is when `lo` > `hi`, otherwise it is a Binary Search.

Then there are two cases:
* if `lo < mid` - monotonic part is on the left
* `lo >= mid` - monotonic part is on the right

Check the monotonic part immediately, otherwise go to the other part.

#### Approach

For more robust code:
* inclusive `lo` and `hi`
* check for target `target == nums[mid]`
* move `lo = mid + 1`, `hi = mid - 1`
* the last case `lo == hi`

#### Complexity

- Time complexity:
$$O(log(n))$$

- Space complexity:
$$O(log(n))$$

#### Code

```kotlin

    fun search(nums: IntArray, target: Int): Int {
      var lo = 0
      var hi = nums.lastIndex
      while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        if (target == nums[mid]) return mid
        if (nums[lo] > nums[hi]) {
          if (nums[lo] > nums[mid]) {
            if (target < nums[mid] || target > nums[hi]) hi = mid - 1 else lo = mid + 1
          } else {
            if (target > nums[mid] || target < nums[lo]) lo = mid + 1 else hi = mid - 1
          }
        } else if (target < nums[mid]) hi = mid - 1 else lo = mid + 1
      }
      return -1
    }

```

