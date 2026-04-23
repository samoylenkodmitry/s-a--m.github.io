---
layout: leetcode-entry
title: "463. Island Perimeter"
permalink: "/leetcode/problem/2024-04-18-463-island-perimeter/"
leetcode_ui: true
entry_slug: "2024-04-18-463-island-perimeter"
---

[463. Island Perimeter](https://leetcode.com/problems/island-perimeter/description/) easy
[blog post](https://leetcode.com/problems/island-perimeter/solutions/5039886/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18042024-463-island-perimeter?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/d91pFLXHb5k)
![2024-04-18_08-48.webp](/assets/leetcode_daily_images/94a4ad97.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/575

#### Problem TLDR

Perimeter of `1`'s islands in `01`-matrix #easy

#### Intuition

Let's observe the problem example:
![2024-04-18_08-05.webp](/assets/leetcode_daily_images/35595899.webp)
As we see, the perimeter increases on the `0`-`1` transitions, we can just count them.
Another neat approach I steal from someone: every `1` increases by 4 and then decreases by `1-1` borders.

#### Approach

Let's try to save some keystrokes
* did you know `compareTo(false)` will convert Boolean to Int? (same is `as i32` in Rust)

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun islandPerimeter(grid: Array<IntArray>) =
        (0..<grid.size * grid[0].size).sumBy { xy ->
            val x = xy % grid[0].size; val y = xy / grid[0].size
            if (grid[y][x] < 1) 0 else
            (x < 1 || grid[y][x - 1] < 1).compareTo(false) +
            (y < 1 || grid[y - 1][x] < 1).compareTo(false) +
            (x == grid[0].lastIndex || grid[y][x + 1] < 1).compareTo(false) +
            (y == grid.lastIndex || grid[y + 1][x] < 1).compareTo(false)
        }

```
```rust

    pub fn island_perimeter(grid: Vec<Vec<i32>>) -> i32 {
        let mut p = 0;
        for y in 0..grid.len() { for x in 0..grid[0].len() {
            if grid[y][x] < 1 { continue }
            if y > 0 && grid[y - 1][x] > 0 { p -= 2 }
            if x > 0 && grid[y][x - 1] > 0 { p -= 2 }
            p += 4
        } }; p
    }

```

