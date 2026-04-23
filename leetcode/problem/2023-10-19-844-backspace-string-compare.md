---
layout: leetcode-entry
title: "844. Backspace String Compare"
permalink: "/leetcode/problem/2023-10-19-844-backspace-string-compare/"
leetcode_ui: true
entry_slug: "2023-10-19-844-backspace-string-compare"
---

[844. Backspace String Compare](https://leetcode.com/problems/backspace-string-compare/description/) medium
[blog post](https://leetcode.com/problems/backspace-string-compare/solutions/4184552/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19102023-844-backspace-string-compare?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/19a77684.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/375

#### Problem TDLR

Are typing with `backspace` sequences equal

#### Intuition

We can use a Stack to evaluate the resulting strings. However, scanning from the end and counting backspaces would work better.

#### Approach

Remove all of the backspaced chars before comparing

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun backspaceCompare(s: String, t: String): Boolean {
      var si = s.lastIndex
      var ti = t.lastIndex
      while (si >= 0 || ti >= 0) {
        var bs = 0
        while (si >= 0 && (s[si] == '#' || bs > 0))
          if (s[si--] == '#') bs++ else bs--
        bs = 0
        while (ti >= 0 && (t[ti] == '#' || bs > 0))
          if (t[ti--] == '#') bs++ else bs--
        if (si < 0 != ti < 0) return false
        if (si >= 0 && s[si--] != t[ti--]) return false
      }
      return true
    }

```

