---
layout: leetcode-entry
title: "861. Score After Flipping Matrix"
permalink: "/leetcode/problem/2024-05-13-861-score-after-flipping-matrix/"
leetcode_ui: true
entry_slug: "2024-05-13-861-score-after-flipping-matrix"
---

[861. Score After Flipping Matrix](https://leetcode.com/problems/score-after-flipping-matrix/description/) medium
[blog post](https://leetcode.com/problems/score-after-flipping-matrix/solutions/5150832/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13052024-861-score-after-flipping?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/yKgQQujHu7M)
![2024-05-13_08-42.webp](/assets/leetcode_daily_images/7fe2d83c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/601

#### Problem TLDR

Max binary-row sum after toggling rows and columns #medium

#### Intuition

Let's consider example:
![2024-05-13_08-10.webp](/assets/leetcode_daily_images/0e909f4d.webp)
Our intuition:
* we can toggle rows only if the `first` bit is `0` otherwise it will make the number smaller
* we can toggle the column only if the number of `0` bits is bigger that `1` bits, otherwise sum will be smaller

#### Approach

We can toggle rows then toggle columns.

* We didn't have to actually toggle columns, just choose the `max(count, height - count)`.
* (The tricky part): we didn't have to toggle rows, just invert each bit if the first bit is zero.

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun matrixScore(grid: Array<IntArray>) =
        grid[0].indices.fold(0) { sum, x ->
            var count = grid.indices.sumOf { grid[it][x] xor grid[it][0] }
            sum * 2 + max(count, grid.size - count)
        }

```
```rust

    pub fn matrix_score(mut grid: Vec<Vec<i32>>) -> i32 {
        (0..grid[0].len()).fold(0, |sum, x| {
            let count: i32 = (0..grid.len()).map(|y| grid[y][0] ^ grid[y][x]).sum();
            sum * 2 + count.max(grid.len() as i32 - count)
        })
    }

```

