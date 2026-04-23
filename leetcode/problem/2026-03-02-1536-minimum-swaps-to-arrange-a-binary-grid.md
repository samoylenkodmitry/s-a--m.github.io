---
layout: leetcode-entry
title: "1536. Minimum Swaps to Arrange a Binary Grid"
permalink: "/leetcode/problem/2026-03-02-1536-minimum-swaps-to-arrange-a-binary-grid/"
leetcode_ui: true
entry_slug: "2026-03-02-1536-minimum-swaps-to-arrange-a-binary-grid"
---

[1536. Minimum Swaps to Arrange a Binary Grid](https://open.substack.com/pub/dmitriisamoilenko/p/02082026-1536-minimum-swaps-to-arrange?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/02082026-1536-minimum-swaps-to-arrange?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02082026-1536-minimum-swaps-to-arrange?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/5PDig8FsghM)

![13f2bfea-b565-4bcf-bae8-9497cdd67bfd (1).webp](/assets/leetcode_daily_images/e88431b2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1285

#### Problem TLDR

Min rows swaps to zero above diagonal #medium

#### Intuition

```j
    // 62% acceptance rate, should be simple
    // 200 nxn=40000
    // count suffix zeros
    // 0
    // 1
    // 2
    // target is
    // n-1
    // n-2
    // ...
    // 1
    // 0
    // or more
    // 012 210
    // how to rearrange optimally?
    // 201
    // 210
    //
    // 022 220
    //
    // 021 210
    //
    // only adjucent means its a bubble sort only
    //
```

* count suffixes
* bubble rows up

#### Approach

* we don't have to add them back

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 22ms
    fun minSwaps(g: Array<IntArray>) =
        ArrayList(g.map { it.lastIndexOf(1) }).run {
            indices.sumOf { i ->
                val j = indices.find { get(it) <= i }
                removeAt(j ?: return -1); j
            }
        }
```
```rust
// 0ms
    pub fn min_swaps(g: Vec<Vec<i32>>) -> i32 {
        let mut s = g.into_iter().map(|r|r.into_iter().rposition(|x|x>0).unwrap_or(0)).collect_vec();
        (0..s.len()).try_fold(0, |r, i| {
            s.iter().position(|&x| x <= i).map(|j|{ s.remove(j); r+j as i32})}).unwrap_or(-1)
    }
```

