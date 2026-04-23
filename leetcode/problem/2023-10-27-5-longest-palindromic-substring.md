---
layout: leetcode-entry
title: "5. Longest Palindromic Substring"
permalink: "/leetcode/problem/2023-10-27-5-longest-palindromic-substring/"
leetcode_ui: true
entry_slug: "2023-10-27-5-longest-palindromic-substring"
---

[5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/description/) medium
[blog post](https://leetcode.com/problems/longest-palindromic-substring/solutions/4212765/kotlin-dp/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27102023-5-longest-palindromic-substring?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/966dc183.webp)
Golf version
![image.png](/assets/leetcode_daily_images/6e7f4b69.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/383

#### Problem TLDR

Longest palindrome substring

#### Intuition

If `dp[from][to]` answering whether substring `s(from, to)` is a palindrome, then `dp[from][to] = s[from] == s[to] && dp[from + 1][to - 1]`

#### Approach

* We can cleverly initialize the `dp` array to avoid some corner cases checks.
* It is better to store just two indices. For simplicity, let's just do `substring` each time.

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun longestPalindrome(s: String): String {
      val dp = Array(s.length) { i -> BooleanArray(s.length) { i >= it } }
      var res = s.take(1)
      for (to in s.indices) for (from in to - 1 downTo 0) {
        dp[from][to] = s[from] == s[to] && dp[from + 1][to - 1]
        if (dp[from][to] && to - from + 1 > res.length)
          res = s.substring(from, to + 1)
      }
      return res
    }

```

