---
layout: leetcode-entry
title: "1984. Minimum Difference Between Highest and Lowest of K Scores"
permalink: "/leetcode/problem/2026-01-25-1984-minimum-difference-between-highest-and-lowest-of-k-scores/"
leetcode_ui: true
entry_slug: "2026-01-25-1984-minimum-difference-between-highest-and-lowest-of-k-scores"
---

[1984. Minimum Difference Between Highest and Lowest of K Scores](https://leetcode.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores/description) easy
[blog post](https://leetcode.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores/solutions/7523215/kotlin-rust-by-samoylenkodmitry-ybup/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25012026-1984-minimum-difference?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/N1rGlao8a8o)

![92716dbf-5bfe-4f84-a465-17644c73b00b (1).webp](/assets/leetcode_daily_images/598f3a5e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1248

#### Problem TLDR

Min k-diff #easy #sliding_window

#### Intuition

Sort, then do a sliding window.

#### Approach

* use windows(k) or zip(skip(k))

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$ or O(1)

#### Code

```kotlin
// 18ms
    fun minimumDifference(n: IntArray, k: Int) =
        n.run{sort();(k..size).minOfOrNull{n[it-1]-n[it-k]}?:0}
```
```rust
// 1ms
    pub fn minimum_difference(mut n: Vec<i32>, k: i32) -> i32 {
       n.sort(); n.iter().zip(n.iter().skip(k as usize-1)).map(|(a,b)|b-a).min().unwrap_or(0)
    }
```

