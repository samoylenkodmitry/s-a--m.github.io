---
layout: leetcode-entry
title: "2390. Removing Stars From a String"
permalink: "/leetcode/problem/2023-04-11-2390-removing-stars-from-a-string/"
leetcode_ui: true
entry_slug: "2023-04-11-2390-removing-stars-from-a-string"
---

[2390. Removing Stars From a String](https://leetcode.com/problems/removing-stars-from-a-string/description/) medium

```kotlin

fun removeStars(s: String): String = StringBuilder().apply {
    s.forEach {
        if (it == '*') setLength(length - 1)
        else append(it)
    }
}.toString()

```

[blog post](https://leetcode.com/problems/removing-stars-from-a-string/solutions/3402891/kotlin-stack/)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/177
#### Intuition
Iterate over a string. When `*` symbol met, remove last character, otherwise add it.
#### Approach
* we can use a `Stack`, or just `StringBuilder`
#### Complexity
- Time complexity:

$$O(n)$$

- Space complexity:

$$O(n)$$

