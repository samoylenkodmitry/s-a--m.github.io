---
layout: leetcode-entry
title: "2215. Find the Difference of Two Arrays"
permalink: "/leetcode/problem/2023-05-03-2215-find-the-difference-of-two-arrays/"
leetcode_ui: true
entry_slug: "2023-05-03-2215-find-the-difference-of-two-arrays"
---

[2215. Find the Difference of Two Arrays](https://leetcode.com/problems/find-the-difference-of-two-arrays/description/) easy

```kotlin

fun findDifference(nums1: IntArray, nums2: IntArray): List<List<Int>> = listOf(
    nums1.subtract(nums2.toSet()).toList(),
    nums2.subtract(nums1.toSet()).toList()
    )

```

[blog post](https://leetcode.com/problems/find-the-difference-of-two-arrays/solutions/3479943/kotlin-one-liner/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-3052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/201
#### Intuition
Just do what is asked.

#### Approach
One way is to use two `Sets` and just filter them.
Another is to use `intersect` and `distinct`.
Third option is to sort both of them and iterate, that will use $$O(1)$$ extra memory, but $$O(nlogn)$$ time.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

