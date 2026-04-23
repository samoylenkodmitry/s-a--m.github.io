---
layout: leetcode-entry
title: "215. Kth Largest Element in an Array"
permalink: "/leetcode/problem/2023-08-14-215-kth-largest-element-in-an-array/"
leetcode_ui: true
entry_slug: "2023-08-14-215-kth-largest-element-in-an-array"
---

[215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/description/) medium
[blog post](https://leetcode.com/problems/kth-largest-element-in-an-array/solutions/3906841/kotlin-quickselect/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14082023-215-kth-largest-element?utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/6750a6f2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/308

#### Problem TLDR

Kth largest in an array

#### Intuition

There is a known Quckselect algorithm:
* do a partition, get the `pivot`
* if `pivot` is less than `target`, repeat on the left side
* otherwise, repeat on the right side of the `pivot`

To do a partition:
* make a growing `buffer` on the left
* choose the `pivot` value which to compare all the elements
* if `nums[i] < pivot`, put and grow the buffer
* finally, put pivot to the end of the buffer
* the buffer size now is a pivot position in a sorted array, as all elements to the left a less than it, and to the right are greater

#### Approach

For divide-and-conquer loop:
* do the last check `from == to`
* always move the border exclusive `from = pi + 1`, `to = pi - 1`

#### Complexity

- Time complexity:
$$O(n) -> O(n^2)$$, the worst case is n^2

- Space complexity:
$$(O(1))$$, but array is modified

#### Code

```kotlin

    fun findKthLargest(nums: IntArray, k: Int): Int {
      var from = 0
      var to = nums.lastIndex
      fun swap(a: Int, b: Int) { nums[a] = nums[b].also { nums[b] = nums[a] } }
      val target = nums.size - k
      while (from <= to) {
        var pi = from
        var pivot = nums[to]
        for (i in from until to) if (nums[i] < pivot) swap(i, pi++)
        swap(to, pi)

        if (pi == target) return nums[pi]
        if (pi < target) from = pi + 1
        if (pi > target) to = pi - 1
      }
      return -1
    }

```

