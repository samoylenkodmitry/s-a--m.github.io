---
layout: leetcode-entry
title: "3871. Count Commas in Range II"
permalink: "/leetcode/problem/2026-09-09-3871-count-commas-in-range-ii/"
leetcode_ui: true
entry_slug: "2026-09-09-3871-count-commas-in-range-ii"
---

[3871. Count Commas in Range II](https://leetcode.com/problems/count-commas-in-range-ii/solutions/8511377/kotlin-rust-by-samoylenkodmitry-0nsh/) medium
[substack](https://dmitriisamoilenko.substack.com/p/09092026-3871-count-commas-in-range?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/MhTeT-IZLsc)

https://dmitrysamoylenko.com/leetcode/

![09.09.2026.webp](/assets/leetcode_daily_images/09.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1477

#### Problem TLDR

Count commas in numbers range

#### Intuition

Number can be big 10^15.
All numbers except first 999 have the first comma, all except frist 999999 has second comma and so on.

#### Approach

* can write a loop

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun countCommas(n: Long) =
    (0..4).sumOf{max(0L,n-"999".repeat(it + 1).toLong())}
```
```rust
    pub fn count_commas(n: i64) -> i64 {
        (1..6).map(|i| (n + 1 - 10i64.pow(i * 3)).max(0)).sum()
    }
```

