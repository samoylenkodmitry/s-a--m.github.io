---
layout: leetcode-entry
title: "1704. Determine if String Halves Are Alike"
permalink: "/leetcode/problem/2022-12-01-1704-determine-if-string-halves-are-alike/"
leetcode_ui: true
entry_slug: "2022-12-01-1704-determine-if-string-halves-are-alike"
---

[1704. Determine if String Halves Are Alike](https://leetcode.com/problems/determine-if-string-halves-are-alike/) easy

[https://t.me/leetcode_daily_unstoppable/38](https://t.me/leetcode_daily_unstoppable/38)

```kotlin

    fun halvesAreAlike(s: String): Boolean {
        val vowels = setOf('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')
        var c1 = 0
        var c2 = 0
        s.forEachIndexed { i, c ->
          if (c in vowels) {
              if (i < s.length / 2) c1++ else c2++
            }
        }
        return c1 == c2
    }

```

Just do what is asked.

O(N) time, O(1) space

