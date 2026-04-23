---
layout: leetcode-entry
title: "77. Combinations"
permalink: "/leetcode/problem/2023-08-01-77-combinations/"
leetcode_ui: true
entry_slug: "2023-08-01-77-combinations"
---

[77. Combinations](https://leetcode.com/problems/combinations/description/) medium
[blog post](https://leetcode.com/problems/combinations/solutions/3845775/kotlin-bitmask/)
[substack](https://dmitriisamoilenko.substack.com/p/01082023-77-combinations?sd=pf)
![image.png](/assets/leetcode_daily_images/3de28ba9.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/295

#### Problem TLDR

All combinations choosing `k` numbers from `1..n` numbers

#### Intuition

As total number is `20`, we can use bit mask to generate all possible `2^n` bit masks, then choose only `k` `1`-bits masks and generate lists.

#### Approach

Let's write a Kotlin one-liner

#### Complexity

- Time complexity:
$$O(n2^n)$$

- Space complexity:
$$O(n2^n)$$

#### Code

```kotlin

    fun combine(n: Int, k: Int): List<List<Int>> = (0 until (1 shl n))
      .filter { Integer.bitCount(it) == k }
      .map { mask -> (1..n).filter { mask and (1 shl it - 1) != 0 } }

```

