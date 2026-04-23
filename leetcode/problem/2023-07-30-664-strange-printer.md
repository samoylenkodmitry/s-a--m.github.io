---
layout: leetcode-entry
title: "664. Strange Printer"
permalink: "/leetcode/problem/2023-07-30-664-strange-printer/"
leetcode_ui: true
entry_slug: "2023-07-30-664-strange-printer"
---

[664. Strange Printer](https://leetcode.com/problems/strange-printer/description/) hard
[blog post](https://leetcode.com/problems/strange-printer/solutions/3836489/kotlin-dp-n-3-find-the-best-split/)
[substack](https://dmitriisamoilenko.substack.com/p/30072023-664-strange-printer?sd=pf)
![image.png](/assets/leetcode_daily_images/caaedf39.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/291

#### Problem TLDR

Minimum continuous overrides by the same character to make a string

#### Intuition

The main idea comes to mind when you consider some `palindromes` as example:

```

abcccba

```

When we consider the next character `ccc + b`, we know, that the optimal number of repaints is `Nc + 1`. Or, `bccc + b`, the optimal is `1 + Nc`.

However, the Dynamic Programming formula for finding a palindrome didn't solve this case: `ababa`, as clearly, the middle `a` can be written in a single path `aaaaa`.

Another idea, is to split the string: `ab + aba`. Number for `ab` = 2, and for `aba` = 2. But, as first == last, we paint `a` only one time, so `dp[from][to] = dp[from][a] + dp[a + 1][to]`.

As we didn't know if our split is the optimal one, we must consider all of them.

#### Approach

* let's write bottom up DP

#### Complexity

- Time complexity:
$$O(n^3)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun strangePrinter(s: String): Int = with(Array(s.length) { IntArray(s.length) }) {
      s.mapIndexed { to, sto ->
        (to downTo 0).map { from -> when {
            to - from <= 1 -> if (s[from] == sto) 1 else 2
            s[from] == sto -> this[from + 1][to]
            else -> (from until to).map { this[from][it] + this[it + 1][to] }.min()!!
          }.also { this[from][to] = it }
        }.last()!!
      }.last()!!
    }

```

