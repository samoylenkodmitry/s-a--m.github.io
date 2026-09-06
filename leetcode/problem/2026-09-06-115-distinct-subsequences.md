---
layout: leetcode-entry
title: "115. Distinct Subsequences"
permalink: "/leetcode/problem/2026-09-06-115-distinct-subsequences/"
leetcode_ui: true
entry_slug: "2026-09-06-115-distinct-subsequences"
---

[115. Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/solutions/8505393/kotlin-rust-by-samoylenkodmitry-s8hk/) hard
[substack](https://dmitriisamoilenko.substack.com/p/06092026-115-distinct-subsequences?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/f5X5JKbAdIU)

https://dmitrysamoylenko.com/leetcode/

![06.09.2026.webp](/assets/leetcode_daily_images/06.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1474

#### Problem TLDR

Count target substrings

#### Intuition

DFS+memo, subproblem starts with suffixes of the string and the target.

#### Approach

* top-down then rewrite to bottom-up

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun numDistinct(s: String, t: String): Int {
        val dp = IntArray(t.length+1); dp[0] = 1
        for (c in s) for (j in t.length - 1 downTo 0)
            if (c == t[j]) dp[j + 1] += dp[j]
        return dp[t.length]
    }
```
```rust
    pub fn num_distinct(s: String, t: String) -> i32 {
        let mut dp = [0; 1001]; dp[0] = 1;
        let (s, t) = (s.as_bytes(), t.as_bytes());
        for &c in s { for j in (0..t.len()).rev() {
            if c == t[j] { dp[j + 1] += dp[j] }
        }} dp[t.len()]
    }
```

