---
layout: leetcode-entry
title: "1930. Unique Length-3 Palindromic Subsequences"
permalink: "/leetcode/problem/2025-11-21-1930-unique-length-3-palindromic-subsequences/"
leetcode_ui: true
entry_slug: "2025-11-21-1930-unique-length-3-palindromic-subsequences"
---

[1930. Unique Length-3 Palindromic Subsequences](https://leetcode.com/problems/unique-length-3-palindromic-subsequences/description/) medium
[blog post](https://leetcode.com/problems/unique-length-3-palindromic-subsequences/solutions/7364368/kotlin-rust-by-samoylenkodmitry-16c9/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21112025-1930-unique-length-3-palindromic?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/pFZatmL5wyY)

![d3163fea-afa0-4a08-9973-cf6a15c7b5d3 (1).webp](/assets/leetcode_daily_images/188d0fa4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1180

#### Problem TLDR

3-palindroms subsequences #medium

#### Intuition

```j
    // how many pairs we have for alphabet?
    // 26*26 - 500*10^5 = 10^7 too big
    // a b c d a  so, between same chars every uniq counts
    // a......b........a.......b  can intersect
```

#### Approach

* we can use bitmask for speedup

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 121ms
    fun countPalindromicSubsequence(s: String) =
    ('a'..'z').sumOf { s.slice(s.indexOf(it)+1..<s.lastIndexOf(it)).toSet().size }
```
```rust
// 127ms
    pub fn count_palindromic_subsequence(s: String) -> i32 {
        ('a'..='z').filter_map(|c| {
            let r = s.rfind(c)?; let l = s[..r].find(c)?;
            Some(s[l+1..r].chars().unique().count() as i32)
        }).sum()
    }
```

