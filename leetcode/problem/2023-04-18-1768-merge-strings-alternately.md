---
layout: leetcode-entry
title: "1768. Merge Strings Alternately"
permalink: "/leetcode/problem/2023-04-18-1768-merge-strings-alternately/"
leetcode_ui: true
entry_slug: "2023-04-18-1768-merge-strings-alternately"
---

[1768. Merge Strings Alternately](https://leetcode.com/problems/merge-strings-alternately/description/) easy

```kotlin

fun mergeAlternately(word1: String, word2: String): String =
(word1.asSequence().zip(word2.asSequence()) { a, b -> "$a$b" } +
word1.drop(word2.length) + word2.drop(word1.length))
.joinToString("")

```

[blog post](https://leetcode.com/problems/merge-strings-alternately/solutions/3429123/kotlin-sequence/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-18052023?sd=pf)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/184
#### Intuition
Do what is asked.
Handle the tail.
#### Approach
* we can use sequence `zip` operator
* for the tail, consider `drop`
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

