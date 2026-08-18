---
layout: leetcode-entry
title: "3471. Find the Largest Almost Missing Integer"
permalink: "/leetcode/problem/2026-08-18-3471-find-the-largest-almost-missing-integer/"
leetcode_ui: true
entry_slug: "2026-08-18-3471-find-the-largest-almost-missing-integer"
---

[3471. Find the Largest Almost Missing Integer](https://leetcode.com/problems/find-the-largest-almost-missing-integer/solutions/8467738/kotlin-rust-by-samoylenkodmitry-3e1d/) easy
[substack](https://dmitriisamoilenko.substack.com/p/18082026-3471-find-the-largest-almost?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/IrnoTDSzGdY)

https://dmitrysamoylenko.com/leetcode/

![18.08.2026.webp](/assets/leetcode_daily_images/18.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1455

#### Problem TLDR

Max number in exactly 1 k-window

#### Intuition

Brute-force lookup all numbers and all windows

#### Approach

* brainteaser O(n) solution possible

#### Complexity

- Time complexity:
$$O(n^3)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun largestInteger(n: IntArray, k: Int) =
    n.filter { n.toList().windowed(k).count { w -> it in w } < 2 }.maxOrNull() ?: -1
```
```rust
    pub fn largest_integer(n: Vec<i32>, k: i32) -> i32 {
        *n.iter().filter(|x|n.windows(k as _).filter(|w|w.contains(x)).count()<2).max().unwrap_or(&-1)
    }
```

