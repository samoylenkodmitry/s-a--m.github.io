---
layout: leetcode-entry
title: "840. Magic Squares In Grid"
permalink: "/leetcode/problem/2024-08-09-840-magic-squares-in-grid/"
leetcode_ui: true
entry_slug: "2024-08-09-840-magic-squares-in-grid"
---

[840. Magic Squares In Grid](https://leetcode.com/problems/magic-squares-in-grid/description/) medium
[blog post](https://leetcode.com/problems/magic-squares-in-grid/solutions/5610738/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09082024-840-magic-squares-in-grid?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/QS6YaYDPBxE)
![1.webp](/assets/leetcode_daily_images/7b923636.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/697

#### Problem TLDR

Count 9x9 1..9 equal row col diag sum subgrids #medium

#### Intuition

Digits must be distinct 1, 2, 3, 4, 5, 6, 7, 8, 9, and all of  them must be present.

#### Approach

Let's just do a brute-force search

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun numMagicSquaresInside(grid: Array<IntArray>): Int {
        var res = 0; val r = -1..1
        for (y in 1..<grid.lastIndex) for (x in 1..<grid[0].lastIndex) {
            if ((0..<9).map { grid[y + it / 3 - 1][x + it % 3 - 1] }
                .filter { it in 1..9 }.toSet().size < 9) continue
            if (setOf(r.sumOf { grid[y - 1][x + it] },
                      r.sumOf { grid[y][x + it] },
                      r.sumOf { grid[y + 1][x + it] },
                      r.sumOf { grid[y + it][x - 1] },
                      r.sumOf { grid[y + it][x] },
                      r.sumOf { grid[y + it][x + 1] },
                      r.sumOf { grid[y + it][x + it] },
                      r.sumOf { grid[y + it][x - it] }).size == 1) res++
        }
        return res
    }

```
```rust

    pub fn num_magic_squares_inside(grid: Vec<Vec<i32>>) -> i32 {
        let mut res = 0;
        for y in 1..grid.len() - 1 { for x in 1..grid[0].len() - 1 {
            let nums = (0..9).map(|i| grid[y + i / 3 - 1][x + i % 3 - 1])
                .filter(|&x| 0 < x && x < 10).collect::<HashSet<_>>();
            if nums.len() < 9 { continue }
            let sums = vec![
                (0..3).map(|i| grid[y + i - 1][x - 1]).sum(),
                (0..3).map(|i| grid[y + i - 1][x]).sum(),
                (0..3).map(|i| grid[y + i - 1][x + 1]).sum(),
                (0..3).map(|i| grid[y - 1][x + i - 1]).sum(),
                (0..3).map(|i| grid[y][x + i - 1]).sum(),
                (0..3).map(|i| grid[y + 1][x + i - 1]).sum(),
                (0..3).map(|i| grid[y + i - 1][x + i - 1]).sum(),
                (0..3).map(|i| grid[y + i - 1][x - i + 1]).sum::<i32>(),
            ];
            if sums.iter().collect::<HashSet<_>>().len() == 1 { res += 1 }
        }}; res
    }

```

