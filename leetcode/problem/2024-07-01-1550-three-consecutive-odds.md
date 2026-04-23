---
layout: leetcode-entry
title: "1550. Three Consecutive Odds"
permalink: "/leetcode/problem/2024-07-01-1550-three-consecutive-odds/"
leetcode_ui: true
entry_slug: "2024-07-01-1550-three-consecutive-odds"
---

[1550. Three Consecutive Odds](https://leetcode.com/problems/three-consecutive-odds/description/) easy
[blog post](https://leetcode.com/problems/three-consecutive-odds/solutions/5394159/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/1072024-1550-three-consecutive-odds?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/jnoO7yObgvE)
![2024-07-01_07-12_1.webp](/assets/leetcode_daily_images/f182a5d1.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/656

#### Problem TLDR

Has window of 3 odds? #easy

#### Intuition

Such questions are helping to start with a new language.

#### Approach

Can you make it shorter?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$ for Rust, O(n) for Kotlin, can be O(1) with `asSequence`.

#### Code

```kotlin

    fun threeConsecutiveOdds(arr: IntArray) =
        arr.asList().windowed(3).any { it.all { it % 2 > 0 }}

```
```rust

    pub fn three_consecutive_odds(arr: Vec<i32>) -> bool {
        arr[..].windows(3).any(|w| w.iter().all(|n| n % 2 > 0))
    }

```

