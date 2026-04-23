---
layout: leetcode-entry
title: "2370. Longest Ideal Subsequence"
permalink: "/leetcode/problem/2024-04-25-2370-longest-ideal-subsequence/"
leetcode_ui: true
entry_slug: "2024-04-25-2370-longest-ideal-subsequence"
---

[2370. Longest Ideal Subsequence](https://leetcode.com/problems/longest-ideal-subsequence/description/) medium
[blog post](https://leetcode.com/problems/longest-ideal-subsequence/solutions/5070085/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25042024-2370-longest-ideal-subsequence?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/40N5oBxrGF4)
![2024-04-25_08-26.webp](/assets/leetcode_daily_images/60d5e3b9.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/583

#### Problem TLDR

Max length of less than `k` adjacent subsequence #medium #dynamic_programming

#### Intuition

Examining some examples, we see some properties:
```j
    // acfgbd   k=2
    // a             a
    //  c            ac
    //   f           f
    //    g          fg
    //     b         acb
    //      d        acbd
```
* we must be able to backtrack to the previous subsequences, so this is full search or at least memoization problem
* at particular position, we know the result for the suffix given the starting char, so we know 26 results
* we can memoise it by (pos, char) key

#### Approach

There are some optimizations:
* current result only depends on the next result, so only [26] results are needed
* we can rewrite memoisation recursion with iterative for-loop
* changing the direction of loop is irrelevant, so better iterate forward for cache friendliness
* the clever trick is to consider only adjacent `k` chars and only update the current char

#### Complexity

- Time complexity:
$$O(n)$$, assuming the alphabet size is constant

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun longestIdealString(s: String, k: Int): Int {
        var dp = IntArray(128)
        for (c in s) dp = IntArray(128) { max(
            if (abs(it - c.code) > k) 0
            else 1 + dp[c.code], dp[it]) }
        return dp.max()
    }

```
```rust

    pub fn longest_ideal_string(s: String, k: i32) -> i32 {
        let mut dp = vec![0; 26];
        for b in s.bytes() {
            let lo = ((b - b'a') as usize).saturating_sub(k as usize);
            let hi = ((b - b'a') as usize + k as usize).min(25);
            dp[(b - b'a') as usize] = 1 + (lo..=hi).map(|a| dp[a]).max().unwrap()
        }
        *dp.iter().max().unwrap()
    }

```

