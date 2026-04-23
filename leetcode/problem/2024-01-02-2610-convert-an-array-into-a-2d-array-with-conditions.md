---
layout: leetcode-entry
title: "2610. Convert an Array Into a 2D Array With Conditions"
permalink: "/leetcode/problem/2024-01-02-2610-convert-an-array-into-a-2d-array-with-conditions/"
leetcode_ui: true
entry_slug: "2024-01-02-2610-convert-an-array-into-a-2d-array-with-conditions"
---

[2610. Convert an Array Into a 2D Array With Conditions](https://leetcode.com/problems/convert-an-array-into-a-2d-array-with-conditions/description/) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/2012024-2610-convert-an-array-into?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[substack](https://youtu.be/Itn5jEpPZ4k)
[youtube](https://youtu.be/Itn5jEpPZ4k)
![image.png](https://assets.leetcode.com/users/images/78cf9bd1-967d-4de2-9948-c311f56960b1_1704183026.395581.png

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/459

#### Problem TLDR

Convert numbers array into array of unique number-rows.

#### Intuition

Let's count each kind of number, then use each unique number to build the rows.

#### Approach

Kotlin's API can be helpful:
* groupBy
* mapValues
* buildList

#### Complexity

- Time complexity:
$$O(uf)$$ where, u - number of uniq elements, f - max frequency. Worst case O(n^2): `1 2 3 4 1 1 1 1`, u = n / 2, f = n / 2. This can be improved to O(n) by removing the empty collections from `freq`.

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun findMatrix(nums: IntArray): List<List<Int>> {
    val freq = nums.groupBy { it }
      .mapValues { it.value.toMutableList() }
    return buildList {
      repeat(freq.values.maxOf { it.size }) {
        add(buildList {
          for ((k, v) in freq)
            if (v.isNotEmpty()) add(v.removeLast())
        })
      }
    }
  }

```

