---
layout: leetcode-entry
title: "2078. Two Furthest Houses With Different Colors"
permalink: "/leetcode/problem/2026-04-20-2078-two-furthest-houses-with-different-colors/"
leetcode_ui: true
entry_slug: "2026-04-20-2078-two-furthest-houses-with-different-colors"
---

[2078. Two Furthest Houses With Different Colors](https://leetcode.com/problems/two-furthest-houses-with-different-colors/solutions/8002008/kotlin-rust-by-samoylenkodmitry-b2qs/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20042026-2078-two-furthest-houses?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/S8AKGk8joIw)

![20.04.2026.webp](/assets/leetcode_daily_images/20.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1334

#### Problem TLDR

Max distance between different numbers #easy

#### Intuition

Brute-force.

Clever solution: first or last is guaranteed to be included (if first==last its entire size, if first!=last then there is a third color in-between).

#### Approach

* scan all indices, pick max to first or last
* downsize the max length while prefix and suffix of this length is not good

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 14ms
    fun maxDistance(c: IntArray) =
    c.indices.last { c[0]!=c[it] || c.last()!=c[c.size-1-it] }
```
```rust
// 0ms
    pub fn max_distance(c: Vec<i32>) -> i32 {
       let n=c.len()-1; (0..n+1).rposition(|l|c[0]!=c[l]||c[n]!=c[n-l]).unwrap() as _
    }
```

