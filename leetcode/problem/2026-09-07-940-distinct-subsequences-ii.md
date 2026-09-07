---
layout: leetcode-entry
title: "940. Distinct Subsequences II"
permalink: "/leetcode/problem/2026-09-07-940-distinct-subsequences-ii/"
leetcode_ui: true
entry_slug: "2026-09-07-940-distinct-subsequences-ii"
---

[940. Distinct Subsequences II](https://leetcode.com/problems/distinct-subsequences-ii/solutions/8506923/kotlin-rust-by-samoylenkodmitry-048z/) hard
[substack](https://dmitriisamoilenko.substack.com/p/07092026-940-distinct-subsequences?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/3EZRs_0zUVM)

https://dmitrysamoylenko.com/leetcode/

![07.09.2026.webp](/assets/leetcode_daily_images/07.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1475

#### Problem TLDR

Count uniq substrings

#### Intuition

DFS+memo. Take or skip. Do not take repeating consequent letters.

#### Approach

* top-down then rewrite to bottom-up

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun distinctSubseqII(s: String) = LongArray(26).apply {
        for (c in s) this[c - 'a'] = (sum() + 1) % 1_000_000_007
    }.sum() % 1_000_000_007
```
```rust
    pub fn distinct_subseq_ii(s: String) -> i32 {
        let mut d = [0; 128];
        for b in s.bytes() { d[b as usize] = (d.iter().sum::<u64>() + 1) % 1000000007 }
        (d.iter().sum::<u64>() % 1000000007) as _
    }
```

