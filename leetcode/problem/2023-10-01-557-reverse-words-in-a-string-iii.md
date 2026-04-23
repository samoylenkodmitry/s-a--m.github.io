---
layout: leetcode-entry
title: "557. Reverse Words in a String III"
permalink: "/leetcode/problem/2023-10-01-557-reverse-words-in-a-string-iii/"
leetcode_ui: true
entry_slug: "2023-10-01-557-reverse-words-in-a-string-iii"
---

[557. Reverse Words in a String III](https://leetcode.com/problems/reverse-words-in-a-string-iii/description/) easy
[blog post](https://leetcode.com/problems/reverse-words-in-a-string-iii/solutions/4112200/kotlin-one-liner/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/1102023-557-reverse-words-in-a-string?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/61ee9014.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/356

#### Problem TLDR

Reverse words

#### Intuition

In an interview in-place solution expected. Maintain two pointers, and adjust one until end of word reached. This still takes O(N) space in JVM.

#### Approach

Let's write a one-liner using Kotlin's API

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun reverseWords(s: String) =
      s.reversed().split(" ").reversed().joinToString(" ")

```

