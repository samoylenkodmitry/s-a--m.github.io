---
layout: leetcode-entry
title: "2264. Largest 3-Same-Digit Number in String"
permalink: "/leetcode/problem/2023-12-04-2264-largest-3-same-digit-number-in-string/"
leetcode_ui: true
entry_slug: "2023-12-04-2264-largest-3-same-digit-number-in-string"
---

[2264. Largest 3-Same-Digit Number in String](https://leetcode.com/problems/largest-3-same-digit-number-in-string/description/) easy
[blog post](https://leetcode.com/problems/largest-3-same-digit-number-in-string/solutions/4360039/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04122023-2264-largest-3-same-digit?r=2bam17&utm_campaign=post&utm_medium=web)
[youtube](https://youtu.be/EWZZKljZf_s)
![image.png](/assets/leetcode_daily_images/edeb3a58.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/427

#### Problem TLDR

Largest 3-same-digit number in a string

#### Intuition

There are totally 10 such numbers: `000, 111, ..., 999`.

#### Approach

Let's use Kotlin's API
* [windowed](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.text/windowed.html)
* [filter](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.text/filter.html)
* [maxByOrNull](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.text/max-by-or-null.html)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, can be O(1) with `asSequence()`

#### Code

```kotlin
    fun largestGoodInteger(num: String): String =
      num.windowed(3)
      .filter { it[0] == it[1] && it[0] == it[2] }
      .maxByOrNull { it[0] } ?: ""

```

