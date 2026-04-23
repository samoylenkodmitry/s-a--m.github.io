---
layout: leetcode-entry
title: "392. Is Subsequence"
permalink: "/leetcode/problem/2023-09-22-392-is-subsequence/"
leetcode_ui: true
entry_slug: "2023-09-22-392-is-subsequence"
---

[392. Is Subsequence](https://leetcode.com/problems/is-subsequence/description/) easy
[blog post](https://leetcode.com/problems/is-subsequence/solutions/4074957/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22092023-392-is-subsequence?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/8b4a8878.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/347

#### Problem TLDR

Is string a subsequence of another

#### Intuition

One possible way is to build a Trie, however this problem can be solved just with two pointers.

#### Approach

Iterate over one string and adjust pointer of another.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun isSubsequence(s: String, t: String): Boolean {
      var i = -1
      return !s.any { c ->
        i++
        while (i < t.length && t[i] != c) i++
        i == t.length
      }
    }

```

