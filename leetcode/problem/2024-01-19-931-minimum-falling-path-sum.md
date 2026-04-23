---
layout: leetcode-entry
title: "931. Minimum Falling Path Sum"
permalink: "/leetcode/problem/2024-01-19-931-minimum-falling-path-sum/"
leetcode_ui: true
entry_slug: "2024-01-19-931-minimum-falling-path-sum"
---

[931. Minimum Falling Path Sum](https://leetcode.com/problems/minimum-falling-path-sum/description/) medium
[blog post](https://leetcode.com/problems/minimum-falling-path-sum/solutions/4590963/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19012024-931-minimum-falling-path?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/aYjobt4BIns)
![image.png](/assets/leetcode_daily_images/a52962f7.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/477

#### Problem TLDR

Min sum moving bottom center, left, right in 2D matrix.

#### Intuition

At every cell we must add it value to path plus min of a three direct top cells as they are the only way here.

#### Approach

We can reuse the matrix or better use separate temporal array.

#### Complexity

- Time complexity:
$$O(mn)$$

- Space complexity:
$$O(1)$$, or O(m) to not corrupt the inputs

#### Code

```kotlin

    fun minFallingPathSum(matrix: Array<IntArray>): Int {
        for (y in 1..<matrix.size) for (x in 0..<matrix[0].size)
            matrix[y][x] += (max(0, x - 1)..min(x + 1, matrix[0].lastIndex))
                .minOf { matrix[y - 1][it] }
        return matrix.last().min()
    }

```
```rust

    pub fn min_falling_path_sum(matrix: Vec<Vec<i32>>) -> i32 {
        *matrix.into_iter().reduce(|dp, row|
            row.iter().enumerate().map(|(x, &v)|
                v + dp[x.max(1) - 1..=(x + 1).min(dp.len() - 1)]
                    .iter().min().unwrap()
            ).collect()
        ).unwrap().iter().min().unwrap()
    }

```

