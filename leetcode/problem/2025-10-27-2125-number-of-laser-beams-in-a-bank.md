---
layout: leetcode-entry
title: "2125. Number of Laser Beams in a Bank"
permalink: "/leetcode/problem/2025-10-27-2125-number-of-laser-beams-in-a-bank/"
leetcode_ui: true
entry_slug: "2025-10-27-2125-number-of-laser-beams-in-a-bank"
---

[2125. Number of Laser Beams in a Bank](https://leetcode.com/problems/number-of-laser-beams-in-a-bank/description) medium
[blog post](https://leetcode.com/problems/number-of-laser-beams-in-a-bank/solutions/7304802/kotlin-rust-by-samoylenkodmitry-zy7p/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27102025-2125-number-of-laser-beams?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/WESY-QrWFLU)

![5a71695d-3b4c-4c0b-bbf8-0c85a67b717f (1).webp](/assets/leetcode_daily_images/205aec1e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1155

#### Problem TLDR

Count multiplications between rows #medium

#### Intuition

Total += previous * current (count '1')

#### Approach

* empty rows are irrelevant

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ or O(1)

#### Code

```kotlin
// 56ms
    fun numberOfBeams(b: Array<String>) = b
    .mapNotNull { it.sumOf { it - '0' }.takeIf { it > 0 }}
    .windowed(2).sumOf { it[0]*it[1] }

```
```rust
// 0ms
    pub fn number_of_beams(b: Vec<String>) -> i32 {
        b.iter().map(|s| s.bytes().filter(|&b| b == b'1').count())
         .filter(|&c| c > 0).tuple_windows().map(|(a,b)| (a*b) as i32).sum()
    }

```

