---
layout: leetcode-entry
title: "1578. Minimum Time to Make Rope Colorful"
permalink: "/leetcode/problem/2025-11-03-1578-minimum-time-to-make-rope-colorful/"
leetcode_ui: true
entry_slug: "2025-11-03-1578-minimum-time-to-make-rope-colorful"
---

[1578. Minimum Time to Make Rope Colorful](https://leetcode.com/problems/minimum-time-to-make-rope-colorful/description) medium
[blog post](https://leetcode.com/problems/minimum-time-to-make-rope-colorful/solutions/7323007/kotlin-rust-by-samoylenkodmitry-mudh/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03112025-1578-minimum-time-to-make?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/mc767eo0eGc)

![b366cc18-6d83-467d-985e-e33344442746 (1).webp](/assets/leetcode_daily_images/d50b815d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1162

#### Problem TLDR

Min weighted removals to dedup #medium #greedy

#### Intuition

Scan from left to right, keep only max from islands of duplicates.

#### Approach

* how many extra variables you need?
* add all time at each step, remove max at change
* or, add all sum(window(min))

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 23ms
    fun minCost(c: String, t: IntArray) =
        (1..<c.length).sumOf { i -> if (c[i] != c[i-1]) 0 else
            min(t[i], t[i-1]).also { t[i] = max(t[i], t[i-1]) }
        }

```
```rust
// 3ms
    pub fn min_cost(c: String, t: Vec<i32>) -> i32 {
        c.bytes().zip(t.iter()).chunk_by(|(a,_)| *a).into_iter()
        .map(|(_, c)| {
            let (s,m) = c.fold((0, 0), |(s,m), (_, &t)| (s+t,m.max(t))); s-m
        }).sum()
    }

```

