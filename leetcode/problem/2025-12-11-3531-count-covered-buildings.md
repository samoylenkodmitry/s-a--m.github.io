---
layout: leetcode-entry
title: "3531. Count Covered Buildings"
permalink: "/leetcode/problem/2025-12-11-3531-count-covered-buildings/"
leetcode_ui: true
entry_slug: "2025-12-11-3531-count-covered-buildings"
---

[3531. Count Covered Buildings](https://leetcode.com/problems/count-covered-buildings/description/) medium
[blog post](https://leetcode.com/problems/count-covered-buildings/solutions/7406516/kotlin-rust-by-samoylenkodmitry-0meq/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11122025-3531-count-covered-buildings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/eO-naf6KpeY)

![9328c4f5-787e-4249-9694-cf67f91a8579 (1).webp](/assets/leetcode_daily_images/82d6d0ff.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1200

#### Problem TLDR

Count surrounded dots on XY plane #medium

#### Intuition

```j
    // idea line sweep Y then X, collect in-betweens, intersect them
```
Line sweep: collect lines, sort each line, drop first and last, intersect with orthogonal sweep.

#### Approach

* optimization: instead of collecting, just look max and min on each line.

#### Complexity

- Time complexity:
$$O(nlog(n))$$, or O(n) for min-max

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 463ms
    fun countCoveredBuildings(n: Int, b: Array<IntArray>) =
        listOf(b.indices.groupBy { b[it][0] }.values.map { it.sortedBy { b[it][1] }},
               b.indices.groupBy { b[it][1] }.values.map { it.sortedBy { b[it][0] }})
        .map { it.fold(HashSet<Int>()) { r, t -> r += t.drop(1).dropLast(1); r }}
        .reduce(Set<Int>::intersect).size
```
```rust
// 17ms
    pub fn count_covered_buildings(n: i32, b: Vec<Vec<i32>>) -> i32 {
        let (mut minY, mut maxY) = (vec![100000; n as usize+1], vec![0; n as usize+1]);
        let (mut minX, mut maxX) = (minY.clone(), maxY.clone());
        for b in &b { let (x,y) = (b[0] as usize, b[1] as usize);
            minY[x] = minY[x].min(y); maxY[x] = maxY[x].max(y);
            minX[y] = minX[y].min(x); maxX[y] = maxX[y].max(x);
        }
        b.iter().filter(|&b| { let (x,y) = (b[0] as usize, b[1] as usize);
            minX[y] < x && x < maxX[y] && minY[x] < y && y < maxY[x] }).count() as _
    }
```

