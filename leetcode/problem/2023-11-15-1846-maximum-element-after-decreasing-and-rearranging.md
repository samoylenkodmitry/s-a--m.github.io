---
layout: leetcode-entry
title: "1846. Maximum Element After Decreasing and Rearranging"
permalink: "/leetcode/problem/2023-11-15-1846-maximum-element-after-decreasing-and-rearranging/"
leetcode_ui: true
entry_slug: "2023-11-15-1846-maximum-element-after-decreasing-and-rearranging"
---

[1846. Maximum Element After Decreasing and Rearranging](https://leetcode.com/problems/maximum-element-after-decreasing-and-rearranging/description/) medium
[blog post](https://leetcode.com/problems/maximum-element-after-decreasing-and-rearranging/solutions/4289555/kotlin-priority-queue/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15112023-1846-maximum-element-after?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/a50d3b31.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/405

#### Problem TLDR

Max number from converting array to non decreasing

#### Intuition

First, sort the array. Now, for every missing number, `1 3 5` -> `2` we can take one of the numbers from the highest, `1 2 3`.
We can use a counter and a Priority Queue.
For example:

```kotlin
array:   1 5 100 100 100
counter: 1 2 3   4   5
```

#### Approach

Let's use some Kotlin's sugar:
* with
* asList

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun maximumElementAfterDecrementingAndRearranging(arr: IntArray): Int =
    with(PriorityQueue<Int>().apply { addAll(arr.asList()) }) {
      var max = 0
      while (isNotEmpty()) if (poll() > max) max++
      max
  }

```
Shorter version:
![image.png](/assets/leetcode_daily_images/903da174.webp)

