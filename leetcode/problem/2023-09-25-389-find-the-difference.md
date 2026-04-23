---
layout: leetcode-entry
title: "389. Find the Difference"
permalink: "/leetcode/problem/2023-09-25-389-find-the-difference/"
leetcode_ui: true
entry_slug: "2023-09-25-389-find-the-difference"
---

[389. Find the Difference](https://leetcode.com/problems/find-the-difference/description/) easy
[blog post](https://leetcode.com/problems/find-the-difference/solutions/4087272/kotlin-one-liner/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25092023-389-find-the-difference?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/f5cab064.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/350

#### Problem TLDR

Strings difference by a single char

#### Intuition

We can use frequency map. Or just calculate total sum by Char Int value.

#### Approach

Let's use Kotlin's API `sumBy`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun findTheDifference(s: String, t: String) =
      (t.sumBy { it.toInt() } - s.sumBy { it.toInt() }).toChar()

```

