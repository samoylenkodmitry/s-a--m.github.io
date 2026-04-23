---
layout: leetcode-entry
title: "1930. Unique Length-3 Palindromic Subsequences"
permalink: "/leetcode/problem/2025-01-04-1930-unique-length-3-palindromic-subsequences/"
leetcode_ui: true
entry_slug: "2025-01-04-1930-unique-length-3-palindromic-subsequences"
---

[1930. Unique Length-3 Palindromic Subsequences](https://leetcode.com/problems/unique-length-3-palindromic-subsequences/description/) medium
[blog post](https://leetcode.com/problems/unique-length-3-palindromic-subsequences/solutions/6228724/kotlin-rust-by-samoylenkodmitry-htcr/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04012025-1930-unique-length-3-palindromic?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/I3QXiWEylZk)
[deep-dive](https://notebooklm.google.com/notebook/5d386e3a-3633-46a5-b9b1-87175fb47cb1/audio)
![1.webp](/assets/leetcode_daily_images/a80028e9.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/854

#### Problem TLDR

Count palindromes of length 3 #medium

#### Intuition

Count unique characters between each pair of the same chars

#### Approach

* building a HashSet can be slower then just checking for contains 26 times

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun countPalindromicSubsequence(s: String) =
        ('a'..'z').sumOf { c ->
            val s = s.slice(s.indexOf(c) + 1..< s.lastIndexOf(c))
            ('a'..'z').count { it in s }
        }

```
```rust

    pub fn count_palindromic_subsequence(s: String) -> i32 {
        ('a'..='z').map(|c| {
            let i = s.find(c).unwrap_or(0);
            let j = s.rfind(c).unwrap_or(0);
            if i + 1 >= j { 0 } else
            { ('a'..='z').filter(|&c| s[i+1..j].contains(c)).count() }
        }).sum::<usize>() as i32
    }

```
```c++

    int countPalindromicSubsequence(string s) {
        int f[26] = {}, l[26] = {}, r = 0; fill(f, f+26, INT_MAX);
        for (int i = 0; i < s.size(); ++i)
            f[s[i] - 'a'] = min(f[s[i] - 'a'], i), l[s[i] - 'a'] = i;
        for (int i = 0; i < 26; ++i) if (f[i] < l[i])
            r += unordered_set<char>(begin(s) + f[i] + 1, begin(s) + l[i]).size();
        return r;
    }

```

