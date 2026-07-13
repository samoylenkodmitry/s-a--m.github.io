---
layout: leetcode-entry
title: "1291. Sequential Digits"
permalink: "/leetcode/problem/2026-07-13-1291-sequential-digits/"
leetcode_ui: true
entry_slug: "2026-07-13-1291-sequential-digits"
---

[1291. Sequential Digits](https://leetcode.com/problems/sequential-digits/solutions/8393935/kotlin-rust-by-samoylenkodmitry-2p1q/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13072026-1291-sequential-digits?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/E7T4fvPS9Z8)

https://dmitrysamoylenko.com/leetcode/

![13.07.2026.webp](/assets/leetcode_daily_images/13.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1419

#### Problem TLDR

Increasing digit numbers in a range

#### Intuition

The possible set of numbers is very small. Generate all then filter and sort.

#### Approach

* iterate over lengths, sliding window over string in an inner loop

#### Complexity

- Time complexity:
$$O(log^2(n))$$

- Space complexity:
$$O(log^2(n))$$

#### Code

```kotlin
    fun sequentialDigits(l: Int, h: Int) =
    (2..9).flatMap{"123456789".windowed(it)}.map{it.toInt()}.filter{it in l..h}
```
```rust
    pub fn sequential_digits(l: i32, h: i32) -> Vec<i32> {
        (2..10).flat_map(|w|(1..11-w).map(move |x| (x..x+w).fold(0,|r,t|r*10+t)))
               .filter(|&x|l<=x&&x<=h).collect()
    }
```

