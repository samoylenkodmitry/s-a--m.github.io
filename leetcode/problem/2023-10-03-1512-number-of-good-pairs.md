---
layout: leetcode-entry
title: "1512. Number of Good Pairs"
permalink: "/leetcode/problem/2023-10-03-1512-number-of-good-pairs/"
leetcode_ui: true
entry_slug: "2023-10-03-1512-number-of-good-pairs"
---

[1512. Number of Good Pairs](https://leetcode.com/problems/number-of-good-pairs/description/) easy
[blog post](https://leetcode.com/problems/number-of-good-pairs/solutions/4122513/kotlin-fold/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/3102023-1512-number-of-good-pairs?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/023ccf47.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/358

#### Problem TLDR

Count equal pairs

#### Intuition

The naive N^2 solution will work.
Another idea is to store the number `frequency` so far and add it to the current result.

#### Approach

Let's use Kotlin's API:
* with
* fold

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun numIdenticalPairs(nums: IntArray) = with(IntArray(101)) {
      nums.fold(0) { r, t -> r + this[t].also { this[t]++ } }
    }

```

