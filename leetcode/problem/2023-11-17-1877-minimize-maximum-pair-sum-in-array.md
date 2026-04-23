---
layout: leetcode-entry
title: "1877. Minimize Maximum Pair Sum in Array"
permalink: "/leetcode/problem/2023-11-17-1877-minimize-maximum-pair-sum-in-array/"
leetcode_ui: true
entry_slug: "2023-11-17-1877-minimize-maximum-pair-sum-in-array"
---

[1877. Minimize Maximum Pair Sum in Array](https://leetcode.com/problems/minimize-maximum-pair-sum-in-array/description/) medium
[blog post](https://leetcode.com/problems/minimize-maximum-pair-sum-in-array/solutions/4297218/kotlin-two-pointers/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17112023-1877-minimize-maximum-pair?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/fdf6d16a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/407

#### Problem TLDR

Minimum possible max of array pairs sums

#### Intuition

The optimal construction way is to pair smallest to largest.

#### Approach

We can use two pointers and iteration, let's write non-optimal one-liner however

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$, this solution takes O(n), but can be rewritten

#### Code

```kotlin

    fun minPairSum(nums: IntArray): Int =
      nums.sorted().run {
          zip(asReversed()).maxOf { it.first + it.second }
      }

```

