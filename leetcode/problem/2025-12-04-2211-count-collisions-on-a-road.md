---
layout: leetcode-entry
title: "2211. Count Collisions on a Road"
permalink: "/leetcode/problem/2025-12-04-2211-count-collisions-on-a-road/"
leetcode_ui: true
entry_slug: "2025-12-04-2211-count-collisions-on-a-road"
---

[2211. Count Collisions on a Road](https://leetcode.com/problems/count-collisions-on-a-road/description/) medium
[blog post](https://leetcode.com/problems/count-collisions-on-a-road/solutions/7390958/kotlin-rust-by-samoylenkodmitry-852v/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04102025-2211-count-collisions-on?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/pdIdPSHuL8c)

![95c23113-c2d5-4f60-8fd7-6d1c65dc353a (1).webp](/assets/leetcode_daily_images/c231a0e2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1193

#### Problem TLDR

Collisions R vs L #medium

#### Intuition

Do separate pass for R, then backwards pass for L.
Drop counter every time opposite meets.

#### Approach

* notice, only the prefix L and suffix R are excluded.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, can be O(1)

#### Code

```kotlin
// 42ms
    fun countCollisions(d: String) =
        d.trimStart('L').trimEnd('R').count { it != 'S' }
```
```rust
// 0ms
    pub fn count_collisions(d: String) -> i32 {
       d.trim_start_matches('L').trim_end_matches('R').bytes()
       .filter(|&b| b != b'S').count() as _
    }
```

