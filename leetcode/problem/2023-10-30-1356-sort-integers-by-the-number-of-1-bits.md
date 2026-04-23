---
layout: leetcode-entry
title: "1356. Sort Integers by The Number of 1 Bits"
permalink: "/leetcode/problem/2023-10-30-1356-sort-integers-by-the-number-of-1-bits/"
leetcode_ui: true
entry_slug: "2023-10-30-1356-sort-integers-by-the-number-of-1-bits"
---

[1356. Sort Integers by The Number of 1 Bits](https://leetcode.com/problems/sort-integers-by-the-number-of-1-bits/description/) easy
[blog post](https://leetcode.com/problems/sort-integers-by-the-number-of-1-bits/solutions/4224952/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30102023-1356-sort-integers-by-the?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/daceb382.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/386

#### Problem TLDR

Sort an array comparing by bit count and value

#### Intuition

Let's use some Kotlin API

#### Approach

* `countOneBits`
* `sortedWith`
* `compareBy`

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun sortByBits(arr: IntArray): IntArray = arr
      .sortedWith(compareBy({ it.countOneBits() }, { it }))
      .toIntArray()

```

