---
layout: leetcode-entry
title: "3546. Equal Sum Grid Partition I"
permalink: "/leetcode/problem/2026-03-25-3546-equal-sum-grid-partition-i/"
leetcode_ui: true
entry_slug: "2026-03-25-3546-equal-sum-grid-partition-i"
---

[3546. Equal Sum Grid Partition I]() medium
[blog post]()
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25032026-3546-equal-sum-grid-partition?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/5fycqFzCLD4)

![25032026.webp](/assets/leetcode_daily_images/25.03.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1308

#### Problem TLDR

Split matrix to equal sums #medium

#### Intuition

O(1) memory solution: calculate total, then separately check each row/column splits p+p==t, sum prefixes in a single variable

Single pass solution: use a hashset to keep track of all visited prefixes, at the end lookup for total/2

#### Approach

* careful with int overflow
* Kotlin: any, sumOf
* Rust: flatten, any, fold

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 24ms
    fun canPartitionGrid(g: Array<IntArray>) =
    g.sumOf { it.sumOf { 1L * it }}.let { t ->
        var h = 0L; var v = 0L
        g.any { h += it.sumOf { 1L * it }; h == t - h } ||
        g[0].indices.any { x ->  v += g.sumOf { 1L * it[x] }; v == t - v }
    }
```
```rust
// 2ms
    pub fn can_partition_grid(g: Vec<Vec<i32>>) -> bool {
        let (mut h, mut v, t) = (0, 0, g.iter().flatten().fold(0, |s,&x| s + x as i64));
        g.iter().any(|r| { h += r.iter().sum::<i32>() as i64; h+h==t}) ||
        (0..g[0].len()).any(|c| { v += g.iter().fold(0,|s,r|s+r[c] as i64); v+v==t})
    }
```

