---
layout: leetcode-entry
title: "1503. Last Moment Before All Ants Fall Out of a Plank"
permalink: "/leetcode/problem/2023-11-04-1503-last-moment-before-all-ants-fall-out-of-a-plank/"
leetcode_ui: true
entry_slug: "2023-11-04-1503-last-moment-before-all-ants-fall-out-of-a-plank"
---

[1503. Last Moment Before All Ants Fall Out of a Plank](https://leetcode.com/problems/last-moment-before-all-ants-fall-out-of-a-plank/description/) medium
[blog post](https://leetcode.com/problems/last-moment-before-all-ants-fall-out-of-a-plank/solutions/4246680/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04112023-1503-last-moment-before?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/39ad85a7.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/392

#### Problem TLDR

Max time ants on a line when goint left and right

#### Intuition

Use the hint: ants can pass through

#### Approach

The problem becomes trivial

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun getLastMoment(n: Int, left: IntArray, right: IntArray): Int =
       max(left.maxOrNull() ?: 0, n - (right.minOrNull() ?: n))

```

