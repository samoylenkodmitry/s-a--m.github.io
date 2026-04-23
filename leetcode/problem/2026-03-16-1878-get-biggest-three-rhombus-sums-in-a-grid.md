---
layout: leetcode-entry
title: "1878. Get Biggest Three Rhombus Sums in a Grid"
permalink: "/leetcode/problem/2026-03-16-1878-get-biggest-three-rhombus-sums-in-a-grid/"
leetcode_ui: true
entry_slug: "2026-03-16-1878-get-biggest-three-rhombus-sums-in-a-grid"
---

[1878. Get Biggest Three Rhombus Sums in a Grid](https://open.substack.com/pub/dmitriisamoilenko/p/16032026-1878-get-biggest-three-rhombus?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/16032026-1878-get-biggest-three-rhombus?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16032026-1878-get-biggest-three-rhombus?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/1wTfpCtE428)

![6e2cf9bd-9a75-4740-9c8a-9e4299b41f05 (1).webp](/assets/leetcode_daily_images/48e8339e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1299

#### Problem TLDR

Top 3 rhombus sums #medium

#### Intuition

Brute-force.

#### Approach

* the distance from center to top is the exact number of steps from top to right

#### Complexity

- Time complexity:
$$O((mn)^2)$$

- Space complexity:
$$O((mn)^2)$$, can be O(1)

#### Code

```kotlin
// 245ms
    fun getBiggestThree(g: Array<IntArray>) = buildSet {
        for (y in g.indices)  for (x in g[0].indices) {
            add(g[y][x])
            for (d in 1..minOf(x, g[0].size-1-x, y, g.size-1-y)) {
                var c = 0; var i = y-d; var j = x
                for ((dy, dx) in listOf(1 to 1, 1 to -1, -1 to -1, -1 to 1))
                    repeat(d) { c += g[i][j]; i += dy; j += dx }
                add(c)
            }
        }
    }.sortedDescending().take(3)
```
```rust
// 13ms
    pub fn get_biggest_three(g: Vec<Vec<i32>>) -> Vec<i32> {
        let (mut s, m, n) = (vec![], g.len(), g[0].len());
        for y in 0..m { for x in 0..n { s.push(g[y][x]);
            for d in 1..=x.min(n-1-x).min(y).min(m-1-y) {
                let (mut c, mut i, mut j) = (0, y-d, x);
                for (dx,dy) in [(1,1),(1,!0),(!0,!0),(!0,1)] { for _ in 0..d
                    { c+=g[i][j]; i=i.wrapping_add(dx); j=j.wrapping_add(dy) }
                }
                s.push(c)
            }
        }} s.sort_unstable_by_key(|&x|-x); s.dedup(); s.truncate(3); s
    }
```

