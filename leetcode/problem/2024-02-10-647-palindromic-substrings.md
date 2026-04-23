---
layout: leetcode-entry
title: "647. Palindromic Substrings"
permalink: "/leetcode/problem/2024-02-10-647-palindromic-substrings/"
leetcode_ui: true
entry_slug: "2024-02-10-647-palindromic-substrings"
---

[647. Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/description/) medium
[blog post](https://leetcode.com/problems/palindromic-substrings/solutions/4704692/kotiln-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10022024-647-palindromic-substrings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/G3yH91q9UQw)
![image.png](/assets/leetcode_daily_images/2cb0afc7.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/501

#### Problem TLDR

Count palindromes substrings.

#### Intuition

There are two possible ways to solve this, one is Dynamic Programming, let's observe some examples first:
```bash
  // aba
  // b -> a b a aba
  // abcba
  // a b c b a bcb abcba
  // aaba -> a a b a aa aba
```
Palindrome can be defined as `dp[i][j] = s[i] == s[j] && dp[i - 1][j + 1]`. This takes quadratic space and time.
Other way to solve is to try to expand from each position. This will be more optimal, as it takes O(1) space and possible O(n) time if there is no palindromes in string. The worst case is O(n^2) however.

#### Approach

Can we make code shorter?

* avoid checking the boundaries of dp[] by playing with initial values and indices

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$ or O(1) for the second.

#### Code

```kotlin

  fun countSubstrings(s: String): Int {
    val dp = Array(s.length + 1) { i ->
      BooleanArray(s.length + 1) { i >= it }}
    return s.indices.sumOf { j ->
        (j downTo 0).count { i ->
        s[i] == s[j] && dp[i + 1][j]
        .also { dp[i][j + 1] = it } } }
  }

```
```rust

  pub fn count_substrings(s: String) -> i32 {
    let s = s.as_bytes();
    let c = |mut l: i32, mut r: usize| -> i32 {
      let mut count = 0;
      while l >= 0 && r < s.len() && s[l as usize] == s[r] {
        l -= 1; r += 1; count += 1;
      }
      count
    };
    (0..s.len()).map(|i| c(i as i32, i) + c(i as i32, i + 1)).sum()
  }

```

