---
layout: leetcode-entry
title: "1074. Number of Submatrices That Sum to Target"
permalink: "/leetcode/problem/2024-01-28-1074-number-of-submatrices-that-sum-to-target/"
leetcode_ui: true
entry_slug: "2024-01-28-1074-number-of-submatrices-that-sum-to-target"
---

[1074. Number of Submatrices That Sum to Target](https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/description/) hard
[blog post](https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/solutions/4637569/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28012024-1074-number-of-submatrices?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/UBxxgETl1v0)
![image.png](/assets/leetcode_daily_images/ff286c47.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/486

#### Problem TLDR

Count submatrix target sums.

#### Intuition

Precompute prefix sums, then calculate submatrix sum in O(1).

#### Approach

* use [n+1][m+1] to avoid `if`s
* there are O(n^3) solution exists

#### Complexity

- Time complexity:
$$O(n^4)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

  fun numSubmatrixSumTarget(matrix: Array<IntArray>, target: Int): Int {
    val s = Array(matrix.size + 1) { IntArray(matrix[0].size + 1) }
    return (1..<s.size).sumOf { y -> (1..<s[0].size).sumOf { x ->
      s[y][x] = matrix[y - 1][x - 1] + s[y - 1][x] + s[y][x - 1] - s[y - 1][x - 1]
      (0..<y).sumOf { y1 -> (0..<x).count { x1 ->
        target == s[y][x] - s[y1][x] - s[y][x1] + s[y1][x1]
      }}
    }}
  }

```
```rust

    pub fn num_submatrix_sum_target(matrix: Vec<Vec<i32>>, target: i32) -> i32 {
      let mut s = vec![vec![0; matrix[0].len() + 1]; matrix.len() + 1];
      (1..s.len()).map(|y| (1..s[0].len()).map(|x| {
        s[y][x] = matrix[y - 1][x - 1] + s[y - 1][x] + s[y][x - 1] - s[y - 1][x - 1];
        (0..y).map(|y1| (0..x).filter_map(|x1|
          if target == s[y][x] - s[y1][x] - s[y][x1] + s[y1][x1] { Some(1) } else { None }
        ).count() as i32).sum::<i32>()
      }).sum::<i32>()).sum()
    }

```

