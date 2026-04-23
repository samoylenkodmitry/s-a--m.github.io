---
layout: leetcode-entry
title: "885. Spiral Matrix III"
permalink: "/leetcode/problem/2024-08-08-885-spiral-matrix-iii/"
leetcode_ui: true
entry_slug: "2024-08-08-885-spiral-matrix-iii"
---

[885. Spiral Matrix III](https://leetcode.com/problems/spiral-matrix-iii/description/) medium
[blog post](https://leetcode.com/problems/spiral-matrix-iii/solutions/5605695/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08082024-885-spiral-matrix-iii?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/OF5c4Scj3FI)
![1.webp](/assets/leetcode_daily_images/67ea622a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/696

#### Problem TLDR

2D spiraling order #medium #matrix #simulation

#### Intuition

One way to write this simulation is to walk over an imaginary path and add the items only when the paths are within the matrix.

We can use a `direction` variable and decide on when to rotate and what to do with an `x` and `y`. Or we can manually iterate in a loop over each side. We should keep the side length `d` and increase it on each cycle of the spiral.
![1.webp](/assets/leetcode_daily_images/5bfb4fb9.webp)

#### Approach

Let's implement direction-walk in Kotlin, and loop-walk in Rust.

#### Complexity

- Time complexity:
$$O(rc)$$

- Space complexity:
$$O(rc)$$

#### Code

```kotlin

    fun spiralMatrixIII(rows: Int, cols: Int, rStart: Int, cStart: Int): Array<IntArray> {
        var y = rStart; var x = cStart; val rx = 0..<cols; val ry = 0..<rows
        var dir = 0; var d = 0
        return Array(rows * cols) { i ->  intArrayOf(y, x).also {
            if (i < rows * cols - 1) do { when (dir) {
                0 -> if (x++ == cStart + d) { d++; dir++ }
                1 -> if (y == rStart + d) { dir++; x-- } else y++
                2 -> if (x == cStart - d) { dir++; y-- } else x--
                3 -> if (y == rStart - d) { dir = 0; x++ } else y--
            }} while (x !in rx || y !in ry)
        }}
    }

```
```rust

    pub fn spiral_matrix_iii(rows: i32, cols: i32, r_start: i32, c_start: i32) -> Vec<Vec<i32>> {
        let (mut y, mut x, mut rx, mut ry) = (r_start, c_start, 0..cols, 0..rows);
        let (mut res, mut d) = (vec![], 1); res.push(vec![y, x]);
        while rows * cols > res.len() as i32 {
            for _ in 0..d { x += 1; if rx.contains(&x) && ry.contains(&y) { res.push(vec![y, x]) }}
            for _ in 0..d { y += 1; if rx.contains(&x) && ry.contains(&y) { res.push(vec![y, x]) }}
            d += 1;
            for _ in 0..d { x -= 1; if rx.contains(&x) && ry.contains(&y) { res.push(vec![y, x]) }}
            for _ in 0..d { y -= 1; if rx.contains(&x) && ry.contains(&y) { res.push(vec![y, x]) }}
            d += 1
        }; res
    }

```

