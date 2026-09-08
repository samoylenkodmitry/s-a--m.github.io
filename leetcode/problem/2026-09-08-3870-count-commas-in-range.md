---
layout: leetcode-entry
title: "3870. Count Commas in Range"
permalink: "/leetcode/problem/2026-09-08-3870-count-commas-in-range/"
leetcode_ui: true
entry_slug: "2026-09-08-3870-count-commas-in-range"
---

[3870. Count Commas in Range](https://leetcode.com/problems/count-commas-in-range/solutions/8508942/kotlin-rust-by-samoylenkodmitry-jgsq/) easy
[substack](https://dmitriisamoilenko.substack.com/p/08092026-3870-count-commas-in-range?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/10T9nQTJ4oY)

https://dmitrysamoylenko.com/leetcode/

![08.09.2026.webp](/assets/leetcode_daily_images/08.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1476

#### Problem TLDR

Count commas in numbers range

#### Intuition

Brute-force is accepted, each number to string and length/3.
Clever way is to spot at most one comma under 10^5 range.

#### Approach

* only the first 999 doesnt have it

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun countCommas(n: Int)=max(0,n-999)
```
```rust
    pub fn count_commas(n: i32) -> i32 { 0.max(n-999) }
```

