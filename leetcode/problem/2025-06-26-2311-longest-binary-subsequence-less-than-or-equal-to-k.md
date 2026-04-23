---
layout: leetcode-entry
title: "2311. Longest Binary Subsequence Less Than or Equal to K"
permalink: "/leetcode/problem/2025-06-26-2311-longest-binary-subsequence-less-than-or-equal-to-k/"
leetcode_ui: true
entry_slug: "2025-06-26-2311-longest-binary-subsequence-less-than-or-equal-to-k"
---

[2311. Longest Binary Subsequence Less Than or Equal to K](https://leetcode.com/problems/longest-binary-subsequence-less-than-or-equal-to-k/description) medium
[blog post](https://leetcode.com/problems/longest-binary-subsequence-less-than-or-equal-to-k/solutions/6887206/kotlin-rust-by-samoylenkodmitry-2v9g/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26062025-2311-longest-binary-subsequence?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2-NEgzxWb5o)
![1.webp](/assets/leetcode_daily_images/bfa9b528.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1031

#### Problem TLDR

Longest binary subsequence less than K #medium

#### Intuition

```j
    // take all zeros
    // 1001010       k=5
    //  ** * *
    //      1        take rightmost 1 while no more than k
    //
    // 1000001110    k=8
    //

    // 1001010     k=5
    //   .   *       l=1
    //   .  *    x=4 l=2
    //   . *         l=3
    //   .-
    //   *           l=4
    //  *            l=5

    // 101001010111100001111110110010011   k=522399436

```
Greedily take from the tail if condition is ok.
Spent too much time trying to build the number, then gave up and just used strings.
(what was missing: check bitshift less than 31)

#### Approach

* sometimes more hacky solution is the only one that can be written without off-by-ones
* to not overflow, check number is not negative, and check the bitshift is less than 31

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 30ms
    fun longestSubsequence(s: String, k: Int) = s.reversed()
        .fold("") { r, c ->
            if ("$c$r".toIntOrNull(2) ?: k + 1 <= k) "$c$r" else r
        }.length

```
```kotlin

// 7ms
    fun longestSubsequence(s: String, k: Int): Int {
        var x = 0; var l = 0
        for (i in s.lastIndex downTo 0)
            if (s[i] == '0') ++l else if (l < 31) {
                val y = x + (1 shl l)
                if (y in 0..k) { x = y; ++l }
            }
        return l
    }

```
```rust

// 0ms
    pub fn longest_subsequence(s: String, k: i32) -> i32 {
        let (mut x, mut l, s) = (0, 0, s.as_bytes());
        for i in (0..s.len()).rev() {
            if s[i] == b'0' { l += 1 }
            else if l < 31 {
                let y = x + (1 << l);
                if y <= k { x = y; l += 1 }
            }
        } l
    }

```
```c++

// 0ms
    int longestSubsequence(string s, int k) {
        int x = 0, l = 0;
        for (int i = size(s) - 1; i >= 0; --i)
            if (s[i] == '0') ++l; else if (l < 31) {
                int y = x + (1 << l);
                if (y <= k) { x = y; ++l; }
            }
        return l;
    }

```

