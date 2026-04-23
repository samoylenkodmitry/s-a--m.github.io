---
layout: leetcode-entry
title: "1219. Path with Maximum Gold"
permalink: "/leetcode/problem/2024-05-14-1219-path-with-maximum-gold/"
leetcode_ui: true
entry_slug: "2024-05-14-1219-path-with-maximum-gold"
---

[1219. Path with Maximum Gold](https://leetcode.com/problems/path-with-maximum-gold/description/) medium
[blog post](https://leetcode.com/problems/path-with-maximum-gold/solutions/5155448/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14052024-1219-path-with-maximum-gold?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/3Py41bk8Xyc)
![2024-05-14_08-57.webp](/assets/leetcode_daily_images/a36b8b34.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/602

#### Problem TLDR

Max one-way path in matrix #medium #dfs

#### Intuition

Path search can almost always be done with a Depth-First Search. Given the problem size `15x15`, we can do a full search with backtracking.

#### Approach

Modify the grid to save some lines of code. Don't do this in a production code however (or document it with warnings).

#### Complexity

- Time complexity:
$$O(3^p)$$, where `p` is the longest path or the number of the gold cells, 3 - is the ways count each step

- Space complexity:
$$O(p)$$, for the recursion depth

#### Code

```kotlin

    fun getMaximumGold(grid: Array<IntArray>): Int {
        fun f(y: Int, x: Int): Int =
            if (grid.getOrNull(y)?.getOrNull(x) ?: 0 < 1) 0 else {
                val v = grid[y][x]; grid[y][x] = 0
                v + maxOf(f(y - 1, x), f(y + 1, x), f(y, x - 1), f(y, x + 1))
                    .also { grid[y][x] = v }
            }
        return grid.indices.maxOf { y -> grid[0].indices.maxOf { f(y, it) }}
    }

```
```rust

    pub fn get_maximum_gold(mut grid: Vec<Vec<i32>>) -> i32 {
        fn f(y: usize, x: usize, grid: &mut Vec<Vec<i32>>) -> i32 {
            let v = grid[y][x]; if v < 1 { return 0 }
            let mut r = 0; grid[y][x] = 0;
            if y > 0 { r = r.max(f(y - 1, x, grid)) }
            if x > 0 { r = r.max(f(y, x - 1, grid)) }
            if y < grid.len() - 1 { r = r.max(f(y + 1, x, grid)) }
            if x < grid[0].len() - 1 { r = r.max(f(y, x + 1, grid)) }
            grid[y][x] = v; r + v
        }
        let mut res = 0;
        for y in 0..grid.len() { for x in 0..grid[0].len() {
            res = res.max(f(y, x, &mut grid))
        }}; res
    }

```

