---
layout: leetcode-entry
title: "799. Champagne Tower"
permalink: "/leetcode/problem/2023-09-24-799-champagne-tower/"
leetcode_ui: true
entry_slug: "2023-09-24-799-champagne-tower"
---

[799. Champagne Tower](https://leetcode.com/problems/champagne-tower/description/) medium
[blog post](https://leetcode.com/problems/champagne-tower/solutions/4083285/kotlin-pascal-s-triangle/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24092023-799-champagne-tower?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/c18f0ba8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/349

#### Problem TLDR

Positional flow value in a Pascal's Triangle

#### Intuition

Let's treat every glass value as the total flow passed through it.
Otherwise, it is a standard Pascal's Triangle problem: reuse the previous row to compute the next.

#### Approach

* if flow is less than `1.0` (full), it will contribute `0.0` to the next row. This can be written as `max(0, x - 1)`
* careful with a champagne, it will beat you in a head

#### Complexity
- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun champagneTower(poured: Int, query_row: Int, query_glass: Int): Double {
      var flow = listOf(poured.toDouble())
      repeat(query_row) {
        val middle = flow.windowed(2).map { (a, b) ->
          max(0.0, a - 1.0) / 2 + max(0.0, b - 1.0) / 2
        }
        val edge = listOf(maxOf(0.0, flow.first() - 1.0) / 2)
        flow = edge + middle + edge
      }
      return minOf(flow[query_glass], 1.0)
    }

```

