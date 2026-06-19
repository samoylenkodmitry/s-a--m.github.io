---
layout: leetcode-entry
title: "1732. Find the Highest Altitude"
permalink: "/leetcode/problem/2026-06-19-1732-find-the-highest-altitude/"
leetcode_ui: true
entry_slug: "2026-06-19-1732-find-the-highest-altitude"
---

[1732. Find the Highest Altitude](https://leetcode.com/problems/find-the-highest-altitude/solutions/8344300/kotlin-rust-by-samoylenkodmitry-88g1/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19062026-1732-find-the-highest-altitude?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/xqogcMssIaQ)

https://dmitrysamoylenko.com/leetcode/

![19.06.2026.webp](/assets/leetcode_daily_images/19.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1395

#### Problem TLDR

Max running sum

#### Intuition

Simulate. Take max.

#### Approach

* Kotlin: scan, Int::plus
* Rust: fold

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1|n)$$

#### Code

```kotlin
    fun largestAltitude(g: IntArray) =
    g.scan(0, Int::plus).max()
```
```rust
    pub fn largest_altitude(g: Vec<i32>) -> i32 {
        g.iter().fold((0,0), |(r,h), x| (r.max(h+x), h+x)).0
    }
```

