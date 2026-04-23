---
layout: leetcode-entry
title: "3567. Minimum Absolute Difference in Sliding Submatrix"
permalink: "/leetcode/problem/2026-03-20-3567-minimum-absolute-difference-in-sliding-submatrix/"
leetcode_ui: true
entry_slug: "2026-03-20-3567-minimum-absolute-difference-in-sliding-submatrix"
---

[3567. Minimum Absolute Difference in Sliding Submatrix](https://open.substack.com/pub/dmitriisamoilenko/p/20032026-3567-minimum-absolute-difference?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/20032026-3567-minimum-absolute-difference?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20032026-3567-minimum-absolute-difference?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/i7A7OTiQO_8)

![bc90b16a-87bd-424d-aef7-2c539e5acf59 (1).webp](/assets/leetcode_daily_images/192c08d7.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1303

#### Problem TLDR

Mid diffs in k*k submatrices #medium #matrix

#### Intuition

Just brute-force, the size is small, 30

#### Approach

* Kotlin: toSortedSet(), windowed
* Rust: itertools allows to use sorted in a single expression, tuple_windows

#### Complexity

- Time complexity:
$$O(nmk^2)$$

- Space complexity:
$$O(nmk^2)$$

#### Code

```kotlin
// 86ms
    fun minAbsDiff(g: Array<IntArray>, k: Int)=
        List(g.size-k+1) { y -> List(g[0].size-k+1) { x ->
            List(k*k){ g[y+it/k][x+it%k] }.toSortedSet()
            .windowed(2) {it[1]-it[0]}.minOrNull() ?: 0
        }}
```
```rust
// 2ms
    pub fn min_abs_diff(g: Vec<Vec<i32>>, k: i32) -> Vec<Vec<i32>> {
        let k = k as usize; (0..=g.len()-k).map(|y|  (0..=g[0].len()-k).map(|x|
            (0..k*k).map(|i| g[y+i/k][x+i%k]).sorted().dedup().tuple_windows()
            .map(|(a,b)|b-a).min().unwrap_or(0)
        ).collect()).collect()
    }
```

