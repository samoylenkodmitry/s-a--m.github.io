---
layout: leetcode-entry
title: "3741. Minimum Distance Between Three Equal Elements II"
permalink: "/leetcode/problem/2026-04-11-3741-minimum-distance-between-three-equal-elements-ii/"
leetcode_ui: true
entry_slug: "2026-04-11-3741-minimum-distance-between-three-equal-elements-ii"
---

[3741. Minimum Distance Between Three Equal Elements II]() medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11042026-3741-minimum-distance-between?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/vmvAUQZt2ao)

![11.04.2026.webp](/assets/leetcode_daily_images/11.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1325

#### Problem TLDR

Min sum of distances between i,j,k of the same values #medium

#### Intuition

Same as yesterday: keep two previous indices. Sum of abs (i,j,k) is 2*(i-k).

```yesterday
     fun minimumDistance(n: IntArray) = n
        .indices.groupBy{n[it]}.values
        .flatMap { it.windowed(3) { 2 * (it[2]-it[0])}}
        .minOrNull() ?: -1

    pub fn minimum_distance(n: Vec<i32>) -> i32 {
        (0..n.len()).into_group_map_by(|&i| n[i]).values()
        .flat_map(|v| v.windows(3).map(|w|2*(w[2]-w[0]) as i32))
        .min().unwrap_or(-1)
    }
```

#### Approach

* let's write optimized versions
* my yesterday golfed solutiona are also accepted

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 85ms
    fun minimumDistance(n: IntArray) =
        LongArray(100001){-1}.let { p -> n.indices.minOf { i ->
            val j = p[n[i]]; p[n[i]] = j shl 32 or 1L*i
            if (j >= 0) 2L*(i-(j shr 32)) else Long.MAX_VALUE
        }.takeIf { it < Long.MAX_VALUE } ?: -1L }
```
```rust
// 16ms
    pub fn minimum_distance(n: Vec<i32>) -> i32 {
        let mut p = [-1i64; 100001];
        (0..n.len()).filter_map(|i| {
            let j = p[n[i] as usize]; p[n[i] as usize] = j<<32|i as i64;
            (j >= 0).then(|| 2*(i as i64 - (j>>32)) as i32)
        }).min().unwrap_or(-1)
    }
```

