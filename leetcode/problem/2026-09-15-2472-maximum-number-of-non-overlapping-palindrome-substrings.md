---
layout: leetcode-entry
title: "2472. Maximum Number of Non-overlapping Palindrome Substrings"
permalink: "/leetcode/problem/2026-09-15-2472-maximum-number-of-non-overlapping-palindrome-substrings/"
leetcode_ui: true
entry_slug: "2026-09-15-2472-maximum-number-of-non-overlapping-palindrome-substrings"
---

[2472. Maximum Number of Non-overlapping Palindrome Substrings](https://leetcode.com/problems/maximum-number-of-non-overlapping-palindrome-substrings/solutions/8522636/kotlin-rust-by-samoylenkodmitry-e6xw/) hard
[substack](https://dmitriisamoilenko.substack.com/p/15092026-2472-maximum-number-of-non?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/_CFL77U-6t0)

https://dmitrysamoylenko.com/leetcode/

![15.09.2026.webp](/assets/leetcode_daily_images/15.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1483

#### Problem TLDR

Max palindrome substrings at least k length

#### Intuition

Greedily choose the smallest palindrome ending with current position.

#### Approach

* can be done recursively
* possible sizes can be just two:  k and k+1

#### Complexity

- Time complexity:
$$O(nk)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun maxPalindromes(s: String, k: Int): Int =
        s.indices.find { i ->
            (max(0,i-k)..i-k+1).any {l->(0..k/2).all{s[l+it]==s[i-it]}}
        }?.let { 1 + maxPalindromes(s.drop(it + 1), k) } ?: 0
```
```rust
    pub fn max_palindromes(s: String, k: i32) -> i32 {
        let u = k as usize; (0..s.len())
            .find(|&i| (i..=i + 1).any(|x| x >= u && s[x - u..=i].bytes().eq(s[x - u..=i].bytes().rev())))
            .map_or(0, |i| 1 + Self::max_palindromes(s[i + 1..].into(), k))
    }
```

