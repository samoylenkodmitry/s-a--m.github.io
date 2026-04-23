---
layout: leetcode-entry
title: "2515. Shortest Distance to Target String in a Circular Array"
permalink: "/leetcode/problem/2026-04-15-2515-shortest-distance-to-target-string-in-a-circular-array/"
leetcode_ui: true
entry_slug: "2026-04-15-2515-shortest-distance-to-target-string-in-a-circular-array"
---

[2515. Shortest Distance to Target String in a Circular Array](https://leetcode.com/problems/shortest-distance-to-target-string-in-a-circular-array/solutions/7916324/kotlin-rust-by-samoylenkodmitry-9lqk/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15042026-2515-shortest-distance-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/BEs5teM31ac)

![15.04.2026.webp](/assets/leetcode_daily_images/15.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1329

#### Problem TLDR

Distance to target #easy

#### Intuition

* (s+dist)%n forward
* (s-dist)%n backward

#### Approach

* don't forget s
* Kotlin has `.mod` that handles sign, indexOfFirst vs firstOrNull
* Rust: `find`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 14ms
    fun closestTarget(w: Array<String>, t: String, s: Int) =
    w.indices.indexOfFirst { w[(s + it)%w.size]==t || w[(s-it).mod(w.size)]==t }
```
```rust
// 0ms
    pub fn closest_target(w: Vec<String>, t: String, s: i32) -> i32 {
        let n = w.len() as i32;
        (0..n).find(|i| w[((s + i) % n) as usize] == t || w[((s - i + n) % n) as usize] == t).unwrap_or(-1)
    }
```

