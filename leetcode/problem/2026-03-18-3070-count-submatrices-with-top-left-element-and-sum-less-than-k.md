---
layout: leetcode-entry
title: "3070. Count Submatrices with Top-Left Element and Sum Less Than k"
permalink: "/leetcode/problem/2026-03-18-3070-count-submatrices-with-top-left-element-and-sum-less-than-k/"
leetcode_ui: true
entry_slug: "2026-03-18-3070-count-submatrices-with-top-left-element-and-sum-less-than-k"
---

[3070. Count Submatrices with Top-Left Element and Sum Less Than k](https://open.substack.com/pub/dmitriisamoilenko/p/18032026-3070-count-submatrices-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/18032026-3070-count-submatrices-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18032026-3070-count-submatrices-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/e8W7RyJi0io)

![c5dfe7df-4595-411f-8b72-d0b56c7e85a3 (1).webp](/assets/leetcode_daily_images/35b1d40f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1301

#### Problem TLDR

Count prefix sums less than K #medium #matrix

#### Intuition

Just compute and re-use prefix sum.

#### Approach

* count in-place
* use row-sum extra variable
* rust itertools have product!
* optimization: after sum bigger than k consider this as the limit for x

#### Complexity

- Time complexity:
$$O(mn)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 39ms
    fun countSubmatrices(g: Array<IntArray>, k: Int) =
        g.indices.sumOf { y -> g[0].indices.count { x ->
            g[y][x] += (if (y > 0) g[y-1][x] else 0) +
                        (if (x > 0) g[y][x-1] else 0) -
                        (if (x*y > 0) g[y-1][x-1] else 0)
            g[y][x] <= k
        }}
```
```rust
// 0ms
    pub fn count_submatrices(mut g: Vec<Vec<i32>>, k: i32) -> i32 {
        (0..g.len()).map(|y| (0..g[0].len()).fold((0,0), |(mut s, c), x| {
            s += g[y][x]; g[y][x] = s + (y>0) as i32 * g[y.max(1)-1][x];
            (s, c + (g[y][x] <= k) as i32)
        }).1).sum()
    }
```

