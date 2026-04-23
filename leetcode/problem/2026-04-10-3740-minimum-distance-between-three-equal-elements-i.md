---
layout: leetcode-entry
title: "3740. Minimum Distance Between Three Equal Elements I"
permalink: "/leetcode/problem/2026-04-10-3740-minimum-distance-between-three-equal-elements-i/"
leetcode_ui: true
entry_slug: "2026-04-10-3740-minimum-distance-between-three-equal-elements-i"
---

[3740. Minimum Distance Between Three Equal Elements I]() easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10042026-3740-minimum-distance-between?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/jABVQm0rxwA)

![10.04.2026.webp](/assets/leetcode_daily_images/10.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1324

#### Problem TLDR

Min sum of distances between i,j,k of the same values #easy

#### Intuition

Brute-force.

#### Approach

* 2*(k-i) is the math optimization
* [101] is the space optimization

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 32ms
    fun minimumDistance(n: IntArray) = n
        .indices.groupBy{n[it]}.values
        .flatMap { it.windowed(3) { 2 * (it[2]-it[0])}}
        .minOrNull() ?: -1
```
```rust
// 0ms
    pub fn minimum_distance(n: Vec<i32>) -> i32 {
        (0..n.len()).into_group_map_by(|&i| n[i]).values()
        .flat_map(|v| v.windows(3).map(|w|2*(w[2]-w[0]) as i32))
        .min().unwrap_or(-1)
    }
```

