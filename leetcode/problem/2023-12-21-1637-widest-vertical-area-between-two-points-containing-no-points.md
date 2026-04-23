---
layout: leetcode-entry
title: "1637. Widest Vertical Area Between Two Points Containing No Points"
permalink: "/leetcode/problem/2023-12-21-1637-widest-vertical-area-between-two-points-containing-no-points/"
leetcode_ui: true
entry_slug: "2023-12-21-1637-widest-vertical-area-between-two-points-containing-no-points"
---

[1637. Widest Vertical Area Between Two Points Containing No Points](https://leetcode.com/problems/widest-vertical-area-between-two-points-containing-no-points/description/) easy
[blog post](https://leetcode.com/problems/widest-vertical-area-between-two-points-containing-no-points/solutions/4434526/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21122023-1637-widest-vertical-area?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
![image.png](/assets/leetcode_daily_images/dee9917b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/446

#### Problem TLDR

Max x window between xy points

#### Intuition

We can sort points by `x` and scan max window between them

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun maxWidthOfVerticalArea(points: Array<IntArray>): Int =
    points
      .sortedBy { it[0] }
      .windowed(2)
      .maxOf { it[1][0] - it[0][0] }

```

