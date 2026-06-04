---
layout: leetcode-entry
title: "3751. Total Waviness of Numbers in Range I"
permalink: "/leetcode/problem/2026-06-04-3751-total-waviness-of-numbers-in-range-i/"
leetcode_ui: true
entry_slug: "2026-06-04-3751-total-waviness-of-numbers-in-range-i"
---

[3751. Total Waviness of Numbers in Range I](https://leetcode.com/problems/total-waviness-of-numbers-in-range-i/solutions/8312576/kotlin-rust-by-samoylenkodmitry-mi3k/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04062026-3751-total-waviness-of-numbers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/_mxzQ39Vwwo)

https://dmitrysamoylenko.com/leetcode/

![04.06.2026.webp](/assets/leetcode_daily_images/04.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1380

#### Problem TLDR

Count hills and valleys base10 in a..b

#### Intuition

Brute-force

#### Approach

* Rust: itertools allows for tuple_windows

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(logn)$$

#### Code

```kotlin
    fun totalWaviness(a: Int, b: Int) =
    (a..b).sumOf {"$it".windowed(3).count {(it[0]-it[1])*(it[2]-it[1])>0}}
```
```rust
    pub fn total_waviness(a: i32, b: i32) -> i32 {
        (a..=b).map(|x|x.to_string().bytes().tuple_windows()
        .filter(|(a,b,c)|a>b&&c>b||a<b&&c<b).count()as i32).sum::<i32>()
    }
```

