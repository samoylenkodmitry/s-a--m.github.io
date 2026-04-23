---
layout: leetcode-entry
title: "200. Number of Islands"
permalink: "/leetcode/problem/2024-04-19-200-number-of-islands/"
leetcode_ui: true
entry_slug: "2024-04-19-200-number-of-islands"
---

[200. Number of Islands](https://leetcode.com/problems/number-of-islands/description/) medium
[blog post](https://leetcode.com/problems/number-of-islands/solutions/5044098/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19042024-200-number-of-islands?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Z-nJMndRFJ4)
![2024-04-19_07-38.webp](/assets/leetcode_daily_images/0375b75d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/576

#### Problem TLDR

Count `1`-islands in `0-1` a 2D matrix #medium

#### Intuition

Let's visit all the connected `1`'s and mark them somehow to visit only once.
Alternative solution would be using Union-Find, however for such trivial case it is unnecessary.

#### Approach

We can modify the input array to mark visited (don't do this in production code or in interview).

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(1)$$, or O(nm) if we forbidden to modify the grid

#### Code

```kotlin

    fun numIslands(grid: Array<CharArray>): Int {
        fun dfs(y: Int, x: Int): Boolean =
            if (grid[y][x] == '1') {
                grid[y][x] = '0'
                if (x > 0) dfs(y, x - 1)
                if (y > 0) dfs(y - 1, x)
                if (x < grid[0].size - 1) dfs(y, x + 1)
                if (y < grid.size - 1) dfs(y + 1, x)
                true
            } else false
        return (0..<grid.size * grid[0].size).count {
            dfs(it / grid[0].size, it % grid[0].size)
        }
    }

```
```rust

    pub fn num_islands(mut grid: Vec<Vec<char>>) -> i32 {
        fn dfs(grid: &mut Vec<Vec<char>>, y: usize, x: usize) -> i32 {
            if grid[y][x] == '1' {
                grid[y][x] = '0';
                if x > 0 { dfs(grid, y, x - 1); }
                if y > 0 { dfs(grid, y - 1, x); }
                if x < grid[0].len() - 1 { dfs(grid, y, x + 1); }
                if y < grid.len() - 1 { dfs(grid, y + 1, x); }
                1
            } else { 0 }
        }
        (0..grid.len() * grid[0].len()).map(|xy| {
            let x = xy % grid[0].len(); let y = xy / grid[0].len();
            dfs(&mut grid, y as usize, x as usize)
        }).sum()
    }

```

