---
layout: leetcode-entry
title: "4. Median of Two Sorted Arrays"
permalink: "/leetcode/problem/2023-09-21-4-median-of-two-sorted-arrays/"
leetcode_ui: true
entry_slug: "2023-09-21-4-median-of-two-sorted-arrays"
---

[4. Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/description/) hard
[blog post](https://leetcode.com/problems/median-of-two-sorted-arrays/solutions/4071065/kotlin-o-n-two-pointer/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21092023-4-median-of-two-sorted-arrays?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/be2b0257.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/346

#### Problem TLDR

Median in two concatenated sorted arrays

#### Intuition

We already know the target position of the median element in the concatenated array.

There is an approach with Binary Search, but it's harder to come up with in an interview and write correctly.

#### Approach

We can maintain two pointers and increase them one by one until `targetPos` reached.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun findMedianSortedArrays(nums1: IntArray, nums2: IntArray): Double {
      val targetPos = (nums1.size + nums2.size) / 2
      var i = 0
      var j = 0
      var prev = 0
      var curr = 0
      while (i + j <= targetPos) {
        prev = curr
        curr = when {
          i == nums1.size -> nums2[j++]
          j == nums2.size -> nums1[i++]
          nums1[i] <= nums2[j] -> nums1[i++]
          else -> nums2[j++]
        }
      }
      return if ((nums1.size + nums2.size) % 2 == 0)
        (prev + curr) / 2.0
       else
        curr.toDouble()
    }

```

