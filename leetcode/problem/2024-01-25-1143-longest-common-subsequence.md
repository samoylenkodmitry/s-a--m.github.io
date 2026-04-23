---
layout: leetcode-entry
title: "1143. Longest Common Subsequence"
permalink: "/leetcode/problem/2024-01-25-1143-longest-common-subsequence/"
leetcode_ui: true
entry_slug: "2024-01-25-1143-longest-common-subsequence"
---

[1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/description) medium
[blog post](https://leetcode.com/problems/longest-common-subsequence/solutions/4622895/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24012024-1143-longest-common-subsequence?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/UrAPRj1TY_w)
![image.png](/assets/leetcode_daily_images/579e3313.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/483

#### Problem TLDR

Longest common subsequence of two strings.

#### Intuition

We can start from a brute force solution: given the current positions `i` and `j` we take them into common if `text1[i] == text2[j]` or choose between taking from `text1[i]` and `text2[j]` if not. The result will only depend on the current positions, so can be cached. From this, we can rewrite the solution to iterative version.

#### Approach

* use `len + 1` dp size to avoid boundary checks
* forward iteration is faster, but `dp[0][0]` must be the out of boundary value
* `fold` can save us some lines of code
* there is a 1D-memory dp solution exists

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

# Code

```kotlin

  fun longestCommonSubsequence(text1: String, text2: String): Int {
    val dp = Array(text1.length + 1) { IntArray(text2.length + 1) }
    for (i in text1.lastIndex downTo 0)
      for (j in text2.lastIndex downTo 0)
        dp[i][j] = if (text1[i] == text2[j])
          1 + dp[i + 1][j + 1] else
          max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]
  }

```
```rust

  pub fn longest_common_subsequence(text1: String, text2: String) -> i32 {
    let mut dp = vec![vec![0; text2.len() + 1]; text1.len() + 1];
    text1.bytes().enumerate().fold(0, |_, (i, a)|
      text2.bytes().enumerate().fold(0, |r, (j, b)| {
        let l = if a == b { 1 + dp[i][j] } else { dp[i][j + 1].max(r) };
        dp[i + 1][j + 1] = l; l
      })
    )
  }

```

