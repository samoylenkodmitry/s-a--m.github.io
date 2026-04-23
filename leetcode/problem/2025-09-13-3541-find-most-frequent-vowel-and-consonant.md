---
layout: leetcode-entry
title: "3541. Find Most Frequent Vowel and Consonant"
permalink: "/leetcode/problem/2025-09-13-3541-find-most-frequent-vowel-and-consonant/"
leetcode_ui: true
entry_slug: "2025-09-13-3541-find-most-frequent-vowel-and-consonant"
---

[3541. Find Most Frequent Vowel and Consonant](https://leetcode.com/problems/find-most-frequent-vowel-and-consonant/description/) easy
[blog post](https://leetcode.com/problems/find-most-frequent-vowel-and-consonant/solutions/7185210/kotlin-by-samoylenkodmitry-7jsh/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13092025-3541-find-most-frequent?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/B5eDJXNtIhY)

![1.webp](/assets/leetcode_daily_images/1e57e18d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1111

#### Problem TLDR

Max freq vowels + consonants #easy

#### Intuition

Make a frequency array, then find max of vowel and max of consonant.

#### Approach

* can we do a one-liner?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 27ms
    fun maxFreqSum(s: String) = s.partition { it in "aeiou" }.toList()
        .sumOf { it.groupBy { it }.maxOfOrNull { it.value.size } ?: 0 }

```

