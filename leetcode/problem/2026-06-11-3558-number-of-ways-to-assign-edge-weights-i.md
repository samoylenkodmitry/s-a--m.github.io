---
layout: leetcode-entry
title: "3558. Number of Ways to Assign Edge Weights I"
permalink: "/leetcode/problem/2026-06-11-3558-number-of-ways-to-assign-edge-weights-i/"
leetcode_ui: true
entry_slug: "2026-06-11-3558-number-of-ways-to-assign-edge-weights-i"
---

[3558. Number of Ways to Assign Edge Weights I](https://leetcode.com/problems/number-of-ways-to-assign-edge-weights-i/solutions/8327456/kotlin-rust-by-samoylenkodmitry-anzn/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11062026-3558-number-of-ways-to-assign?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/jWSiyPJdCcU)

https://dmitrysamoylenko.com/leetcode/

![11.06.2026.webp](/assets/leetcode_daily_images/11.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1387

#### Problem TLDR

Ways to color the deepest path in tree

#### Intuition

* construct the tree, find the path length
* the number of two-color the path is 2^len

#### Approach

* use groupBy
* modPow is an overkill, fold is enough

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun assignEdgeWeights(e: Array<IntArray>): Long {
        val g = e.flatMap {(a,b) -> setOf(a to b,b to a)}.groupBy({it.first},{it.second})
        fun d(i: Int, p: Int): Int = g[i]?.maxOf { j -> if (j == p) 0 else 1 + d(j, i) }?:0
        return (1..<d(1, 0)).fold(1L){r,_ -> (r*2)%1000000007}
    }
```
```rust
    pub fn assign_edge_weights(e: Vec<Vec<i32>>) -> i32 {
        let g = e.into_iter().flat_map(|v| [(v[0], v[1]), (v[1], v[0])]).into_group_map();
        fn d(i: i32, p: i32, g: &HashMap<i32, Vec<i32>>) -> i32 {
            g.get(&i).into_iter().flatten().map(|&j| if j == p { 0 } else { 1 + d(j, i, g) }).max().unwrap_or(0)
        }
        (1..d(1, 0, &g)).fold(1, |r, _| r * 2 % 1_000_000_007)
    }
```

