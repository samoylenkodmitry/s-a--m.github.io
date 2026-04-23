---
layout: leetcode-entry
title: "1458. Max Dot Product of Two Subsequences"
permalink: "/leetcode/problem/2023-10-08-1458-max-dot-product-of-two-subsequences/"
leetcode_ui: true
entry_slug: "2023-10-08-1458-max-dot-product-of-two-subsequences"
---

[1458. Max Dot Product of Two Subsequences](https://leetcode.com/problems/max-dot-product-of-two-subsequences/description/) hard
[blog post](https://leetcode.com/problems/max-dot-product-of-two-subsequences/solutions/4144292/kotlin-dp/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/8102023-1458-max-dot-product-of-two?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/e6fbe6a8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/363

#### Problem TLDR

Max product of two subsequences

#### Intuition

We can search in all possible subsequences in O(n^2) by choosing between: take element and stop, take and continue, skip first, skip second.

#### Approach

The top-down aproach is trivial, let's modify it into bottom up.
* use sentry `dp` size to avoid writing `if`s

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun maxDotProduct(nums1: IntArray, nums2: IntArray): Int {
      val dp = Array(nums1.size + 1) { Array(nums2.size + 1) { -1000000 } }
      for (j in nums2.lastIndex downTo 0)
        for (i in nums1.lastIndex downTo 0)
          dp[i][j] = maxOf(
              nums1[i] * nums2[j],
              nums1[i] * nums2[j] + dp[i + 1][j + 1],
              dp[i][j + 1],
              dp[i + 1][j])
      return dp[0][0]
    }

```

