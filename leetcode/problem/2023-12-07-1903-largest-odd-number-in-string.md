---
layout: leetcode-entry
title: "1903. Largest Odd Number in String"
permalink: "/leetcode/problem/2023-12-07-1903-largest-odd-number-in-string/"
leetcode_ui: true
entry_slug: "2023-12-07-1903-largest-odd-number-in-string"
---

[1903. Largest Odd Number in String](https://leetcode.com/problems/largest-odd-number-in-string/description/) easy
[blog post](https://leetcode.com/problems/largest-odd-number-in-string/solutions/4374041/kotlin-one-liner/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07122023-1903-largest-odd-number?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/d767053b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/431

#### Problem TLDR

Largest odd number in a string

#### Intuition

Just search for the last odd

#### Approach

Let's write Kotlin one-liner

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun largestOddNumber(num: String): String =
    num.dropLastWhile { it.toInt() % 2 == 0 }

```

