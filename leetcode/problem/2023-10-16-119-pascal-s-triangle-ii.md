---
layout: leetcode-entry
title: "119. Pascal's Triangle II"
permalink: "/leetcode/problem/2023-10-16-119-pascal-s-triangle-ii/"
leetcode_ui: true
entry_slug: "2023-10-16-119-pascal-s-triangle-ii"
---

[119. Pascal's Triangle II](https://leetcode.com/problems/pascals-triangle-ii/description/) easy
[blog post](https://leetcode.com/problems/pascals-triangle-ii/solutions/4173651/kotlin-fold/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16102023-119-pascals-triangle-ii?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/a1301c77.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/372

#### Problem TLDR

Pascal's Triangle

#### Intuition

One way is to generate sequence:

```kotlin
    fun getRow(rowIndex: Int): List<Int> =
      generateSequence(listOf(1)) {
        listOf(1) + it.windowed(2) { it.sum() } + 1
      }.elementAtOrElse(rowIndex) { listOf() }
```

Another way is to use `fold`

#### Approach

* notice, we can add a simple `1` to collection by `+`
* use `sum` and `windowed`

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun getRow(rowIndex: Int): List<Int> =
      (1..rowIndex).fold(listOf(1)) { r, _ ->
        listOf(1) + r.windowed(2) { it.sum() } + 1
      }

```

