---
layout: leetcode-entry
title: "1347. Minimum Number of Steps to Make Two Strings Anagram"
permalink: "/leetcode/problem/2024-01-13-1347-minimum-number-of-steps-to-make-two-strings-anagram/"
leetcode_ui: true
entry_slug: "2024-01-13-1347-minimum-number-of-steps-to-make-two-strings-anagram"
---

[1347. Minimum Number of Steps to Make Two Strings Anagram](https://leetcode.com/problems/minimum-number-of-steps-to-make-two-strings-anagram/description/) medium
[blog post](https://leetcode.com/problems/minimum-number-of-steps-to-make-two-strings-anagram/solutions/4556656/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13012024-1347-minimum-number-of-steps?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/7yGUxNR6cVI)
![image.png](/assets/leetcode_daily_images/20edd9c0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/470

#### Problem TLDR

Min operations to make string `t` anagram of `s`.

#### Intuition

Let's compare char's frequencies of those two strings.

#### Approach

* careful: as we replacing one kind of chars with another, we must decrease that another counter

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun minSteps(s: String, t: String) =
    IntArray(128).let {
      for (c in s) it[c.toInt()]++
      for (c in t) it[c.toInt()]--
      it.sumOf { abs(it) } / 2
    }

```

