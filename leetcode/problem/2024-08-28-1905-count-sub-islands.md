---
layout: leetcode-entry
title: "1905. Count Sub Islands"
permalink: "/leetcode/problem/2024-08-28-1905-count-sub-islands/"
leetcode_ui: true
entry_slug: "2024-08-28-1905-count-sub-islands"
---

[1905. Count Sub Islands](https://leetcode.com/problems/count-sub-islands/description/) medium
[blog post](https://leetcode.com/problems/count-sub-islands/solutions/5701082/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28082024-1905-count-sub-islands?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/kOc13waUpaE)
![1.webp](/assets/leetcode_daily_images/f980e264.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/716

#### Problem TLDR

Count islands intersecting in both 2D grids #medium #dfs

#### Intuition

First, understand the problem: not just intersecting `1` cells, but they must all lie on continuous islands without `0` breaks.
Explore `grid2` islands and filter out if it has `0` in `grid1` in them.

#### Approach

Let's use iterators.
* we can mark visited nodes modifying the grid

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun countSubIslands(grid1: Array<IntArray>, grid2: Array<IntArray>): Int {
        fun dfs(y: Int, x: Int): Boolean = grid2[y][x] == 0 || {
            grid2[y][x] = 0
            (grid1[y][x] == 1) and
            (y == 0 || dfs(y - 1, x)) and
            (x == 0 || dfs(y, x - 1)) and
            (y == grid2.size - 1 || dfs(y + 1, x)) and
            (x == grid2[0].size - 1 || dfs(y, x + 1))
        }()
        return grid2.withIndex().sumOf { (y, r) ->
            r.withIndex().count { (x, c) -> c > 0 && dfs(y, x) }}
    }

```
```rust

    pub fn count_sub_islands(mut grid1: Vec<Vec<i32>>, mut grid2: Vec<Vec<i32>>) -> i32 {
        fn dfs(grid1: &[Vec<i32>], grid2: &mut Vec<Vec<i32>>, y: usize, x: usize) -> bool {
            grid2[y][x] == 0 || {
                grid2[y][x] = 0;
                (grid1[y][x] == 1) &
                (y == 0 || dfs(grid1, grid2, y - 1, x)) &
                (x == 0 || dfs(grid1, grid2, y, x - 1)) &
                (y == grid2.len() - 1 || dfs(grid1, grid2, y + 1, x)) &
                (x == grid2[0].len() - 1 || dfs(grid1, grid2, y, x + 1))
            }}
        let w = grid2[0].len(); (0..grid2.len() * w)
        .filter(|i| grid2[i / w][i % w] > 0 && dfs(&grid1, &mut grid2, i / w, i % w)).count() as i32
    }

```

