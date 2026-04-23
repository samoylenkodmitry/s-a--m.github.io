---
layout: leetcode-entry
title: "1657. Determine if Two Strings Are Close"
permalink: "/leetcode/problem/2024-01-14-1657-determine-if-two-strings-are-close/"
leetcode_ui: true
entry_slug: "2024-01-14-1657-determine-if-two-strings-are-close"
---

[1657. Determine if Two Strings Are Close](https://leetcode.com/problems/determine-if-two-strings-are-close/description/) medium
[blog post](https://leetcode.com/problems/determine-if-two-strings-are-close/solutions/4562444/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14012024-1657-determine-if-two-strings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/enAXGzsmVB8)
![image.png](/assets/leetcode_daily_images/15d3527d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/471

#### Problem TLDR

Are strings convertible by swapping existing chars positions or frequencies.

#### Intuition

By the problem definition, we must compare the frequencies numbers. Also, sets of chars must be equal.

#### Approach

Let's use some Kotlin's API:
* groupingBy
* eachCount
* run
* sorted

#### Complexity

- Time complexity:
$$O(n)$$, as we are sorting only 26 elements

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun String.f() = groupingBy { it }.eachCount()
    .run { keys to values.sorted() }
  fun closeStrings(word1: String, word2: String) =
    word1.f() == word2.f()

```

