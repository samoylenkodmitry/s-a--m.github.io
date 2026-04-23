---
layout: leetcode-entry
title: "118. Pascal's Triangle"
permalink: "/leetcode/problem/2023-09-08-118-pascal-s-triangle/"
leetcode_ui: true
entry_slug: "2023-09-08-118-pascal-s-triangle"
---

[118. Pascal's Triangle](https://leetcode.com/problems/pascals-triangle/description/) easy
[blog post](https://leetcode.com/problems/pascals-triangle/solutions/4016541/kotlin-running-fold/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/8092023-118-pascals-triangle?utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/7891eba4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/333

#### Problem TLDR

Pascal Triangle

#### Intuition

Each row is a previous row sliding window sums concatenated with `1`

#### Approach

Let's write it using Kotlin API

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun generate(numRows: Int) = (2..numRows)
      .runningFold(listOf(1)) { r, _ ->
        listOf(1) + r.windowed(2).map { it.sum() } + listOf(1)
      }

```

