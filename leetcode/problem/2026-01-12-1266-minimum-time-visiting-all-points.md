---
layout: leetcode-entry
title: "1266. Minimum Time Visiting All Points"
permalink: "/leetcode/problem/2026-01-12-1266-minimum-time-visiting-all-points/"
leetcode_ui: true
entry_slug: "2026-01-12-1266-minimum-time-visiting-all-points"
---

[1266. Minimum Time Visiting All Points](https://leetcode.com/problems/minimum-time-visiting-all-points/description) easy
[blog post](https://leetcode.com/problems/minimum-time-visiting-all-points/solutions/7488565/kotlin-rust-by-samoylenkodmitry-0ck3/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12012026-1266-minimum-time-visiting?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ZojOi34kLHY)

![cf961b43-5360-4846-bd81-549443325d86 (1).webp](/assets/leetcode_daily_images/6162a478.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1235

#### Problem TLDR

Euclid travel distance #easy #brainteaser

#### Intuition

D = max(dx,dy)

#### Approach

* indices: [1..n)
* zip, zipWithNext, windows, fold

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 6ms
    fun minTimeToVisitAllPoints(p: Array<IntArray>) = (1..<p.size)
    .sumOf {i->max(abs(p[i][0]-p[i-1][0]), abs(p[i][1]-p[i-1][1]))}
```
```rust
// 0ms
    pub fn min_time_to_visit_all_points(p: Vec<Vec<i32>>) -> i32 {
        p.iter().zip(&p[1..]).map(|(a,b)|(a[0]-b[0]).abs().max((a[1]-b[1]).abs())).sum()
    }
```

