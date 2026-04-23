---
layout: leetcode-entry
title: "1289. Minimum Falling Path Sum II"
permalink: "/leetcode/problem/2024-04-26-1289-minimum-falling-path-sum-ii/"
leetcode_ui: true
entry_slug: "2024-04-26-1289-minimum-falling-path-sum-ii"
---

[1289. Minimum Falling Path Sum II](https://leetcode.com/problems/minimum-falling-path-sum-ii/description/) hard
[blog post](https://leetcode.com/problems/minimum-falling-path-sum-ii/solutions/5073998/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26042024-1289-minimum-falling-path?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/N-zT-RrkSb8)
![2024-04-26_08-15.webp](/assets/leetcode_daily_images/58827ada.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/584

#### Problem TLDR

Min non-direct path top down in a 2D matrix #hard #dynamic_programming

#### Intuition

Let's try an example:
![2024-04-26_07-43.webp](/assets/leetcode_daily_images/bac54078.webp)
On each row we need to know the `min` value from the previous row, or the `second min`, if first is directly up. Then adding this min to the current cell would give us the min-sum.

#### Approach

We can reuse the matrix for brevety, however don't do this in the interview or in a production code.

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(1)$$, or O(m) if the separate array used

#### Code

```kotlin

    fun minFallingPathSum(grid: Array<IntArray>): Int {
        var min1 = -1; var min2 = -1
        for (y in grid.indices) { grid[y].let {
            if (y > 0) for (x in it.indices)
                it[x] += grid[y - 1][if (x == min1) min2 else min1]
            min1 = -1; min2 = -1
            for (x in it.indices)
                if (min1 < 0 || it[x] < it[min1]) {
                    min2 = min1; min1 = x
                } else if (min2 < 0 || it[x] < it[min2]) min2 = x
        }}
        return grid.last()[min1]
    }

```
```rust

    pub fn min_falling_path_sum(mut grid: Vec<Vec<i32>>) -> i32 {
        let n = grid[0].len(); let (mut min1, mut min2) = (n, n);
        for y in 0..grid.len() {
            if y > 0 { for x in 0..n {
                grid[y][x] += grid[y - 1][if x == min1 { min2 } else { min1 }]
            }}
            min1 = n; min2 = n;
            for x in 0..n {
                if min1 == n || grid[y][x] < grid[y][min1] {
                    min2 = min1; min1 = x
                } else if min2 == n || grid[y][x] < grid[y][min2] { min2 = x }
            }
        }
        grid[grid.len() - 1][min1]
    }

```

