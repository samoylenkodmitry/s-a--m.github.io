---
layout: leetcode-entry
title: "3702. Longest Subsequence With Non-Zero Bitwise XOR"
permalink: "/leetcode/problem/2026-08-15-3702-longest-subsequence-with-non-zero-bitwise-xor/"
leetcode_ui: true
entry_slug: "2026-08-15-3702-longest-subsequence-with-non-zero-bitwise-xor"
---

[3702. Longest Subsequence With Non-Zero Bitwise XOR](https://leetcode.com/problems/longest-subsequence-with-non-zero-bitwise-xor/solutions/8462176/kotlin-rust-by-samoylenkodmitry-9lk3/) medium
[substack](https://dmitriisamoilenko.substack.com/p/15082026-3702-longest-subsequence?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/5kbwIxvft1I)

https://dmitrysamoylenko.com/leetcode/

![15.08.2026.webp](/assets/leetcode_daily_images/15.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1452

#### Problem TLDR

Max subsequence to max non-zero xor

#### Intuition

XOR is zero if all values are zero or if they are two equal aubsequencies. To make them non-equal just remove one number.

#### Approach

* or simulate with brute force and look at the results

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun longestSubsequence(n: IntArray) =
        if (n.all{it<1}) 0 else n.size-if (n.fold(0,Int::xor)>0)0 else 1
```
```rust
    pub fn longest_subsequence(n: Vec<i32>) -> i32 {
        (n.iter().any(|&x|x>0)as i32)*(n.len()as i32-(n.iter().fold(0,|a,b|a^b)<1)as i32)
    }
```

