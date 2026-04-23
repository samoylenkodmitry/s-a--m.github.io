---
layout: leetcode-entry
title: "2373. Largest Local Values in a Matrix"
permalink: "/leetcode/problem/2024-05-12-2373-largest-local-values-in-a-matrix/"
leetcode_ui: true
entry_slug: "2024-05-12-2373-largest-local-values-in-a-matrix"
---

[2373. Largest Local Values in a Matrix](https://leetcode.com/problems/largest-local-values-in-a-matrix/description/) easy
[blog post](https://leetcode.com/problems/largest-local-values-in-a-matrix/solutions/5146347/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12052024-2373-largest-local-values?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/_zMW6w9aUoQ)
![2024-05-12_08-45.webp](/assets/leetcode_daily_images/4b627a15.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/600

#### Problem TLDR

Max pooling by `3x3` matrix #easy

#### Intuition

The easiest way is to just iterate over the neighbours each time. (However one can possible find an algorithm to do a running-max with a monotonic stack)

#### Approach

Let's try to write it shorter this time.

#### Complexity

- Time complexity:
$$O(n^2k^4)$$, where k = 3 is constant

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun largestLocal(grid: Array<IntArray>) =
        Array(grid.size - 2) { y -> IntArray(grid.size - 2) { x ->
            (0..8).maxOf { grid[y + it / 3][x + it % 3] }
        }}

```
```rust

    pub fn largest_local(grid: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let mut res = vec![vec![0; grid.len() - 2]; grid.len() - 2];
        for y in 0..res.len() { for x in 0..res.len() {
            res[y][x] = (0..9).map(|i| grid[y + i / 3][x + i % 3]).max().unwrap()
        }}; res
    }

```

