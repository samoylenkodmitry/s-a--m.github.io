---
layout: leetcode-entry
title: "242. Valid Anagram"
permalink: "/leetcode/problem/2023-12-16-242-valid-anagram/"
leetcode_ui: true
entry_slug: "2023-12-16-242-valid-anagram"
---

[242. Valid Anagram](https://leetcode.com/problems/valid-anagram/description/) easy
[blog post](https://leetcode.com/problems/valid-anagram/solutions/4411609/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16122023-242-valid-anagram?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
![image.png](/assets/leetcode_daily_images/405fd7d1.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/440

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, can also be solved in O(1) by computing the `hash`

#### Code

```kotlin

    fun isAnagram(s: String, t: String): Boolean =
      s.groupBy { it } == t.groupBy { it }

```

