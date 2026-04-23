---
layout: leetcode-entry
title: "2125. Number of Laser Beams in a Bank"
permalink: "/leetcode/problem/2024-01-03-2125-number-of-laser-beams-in-a-bank/"
leetcode_ui: true
entry_slug: "2024-01-03-2125-number-of-laser-beams-in-a-bank"
---

[2125. Number of Laser Beams in a Bank](https://leetcode.com/problems/number-of-laser-beams-in-a-bank/description/) medium
[blog post](https://leetcode.com/problems/number-of-laser-beams-in-a-bank/solutions/4496627/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/3012024-2125-number-of-laser-beams?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/h-SRyUsiCZc)
![image.png](/assets/leetcode_daily_images/a7e02a99.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/460

#### Problem TLDR

Beams count between consequent non-empty row's `1`s.

#### Intuition

By the problem definition, `count = sum_i_j(count_i * count_j)`

#### Approach

Let's use some Kotlin's API:
* map
* filter
* windowed
* sum

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(n)$$, can be reduced to O(1) with `asSequence` and `fold`.

#### Code

```kotlin

  fun numberOfBeams(bank: Array<String>) =
    bank.map { it.count { it == '1' } }
      .filter { it > 0 }
      .windowed(2)
      .map { (a, b) -> a * b }
      .sum() ?: 0

```

