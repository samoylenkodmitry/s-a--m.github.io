---
layout: leetcode-entry
title: "1463. Cherry Pickup II"
permalink: "/leetcode/problem/2024-02-11-1463-cherry-pickup-ii/"
leetcode_ui: true
entry_slug: "2024-02-11-1463-cherry-pickup-ii"
---

[1463. Cherry Pickup II](https://leetcode.com/problems/cherry-pickup-ii/description/) medium
[blog post](https://leetcode.com/problems/cherry-pickup-ii/description/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11022024-1463-cherry-pickup-ii?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/jbiDBf5oHXs)

![image.png](/assets/leetcode_daily_images/9103cc00.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/502

#### Problem TLDR

Maximum paths sum of two robots top-down in XY grid.

#### Intuition
One way is to try all possible paths, but that will give TLE.
However, we can notice, that only start position of two robots matters, so result can be cached:
![image.png](/assets/leetcode_daily_images/50fb2196.webp)

Another neat optimization is to forbid to intersect the paths.

#### Approach

Can you make code shorter?
* wrapping_add for Rust
* takeIf, maxOf, in Range for Kotlin

#### Complexity

- Time complexity:
$$O(mn^2)$$

- Space complexity:
$$(mn^2)$$

#### Code

```kotlin

  fun cherryPickup(grid: Array<IntArray>): Int {
    val r = 0..<grid[0].size
    val ways = listOf(-1 to -1, -1 to 0, -1 to 1,
                       0 to -1,  0 to 0,  0 to 1,
                       1 to -1,  1 to 0,  1 to 1)
    val dp = Array(grid.size) {
             Array(grid[0].size) {
             IntArray(grid[0].size) { -1 } } }
    fun dfs(y: Int, x1: Int, x2: Int): Int =
      dp[y][x1][x2].takeIf { it >= 0 } ?: {
        grid[y][x1] + grid[y][x2] +
        if (y == grid.lastIndex) 0 else ways.maxOf { (dx1, dx2) ->
          val nx1 = x1 + dx1
          val nx2 = x2 + dx2
          if (nx1 in r && nx2 in r && nx1 < nx2) { dfs(y + 1, nx1, nx2) } else 0
      }}().also { dp[y][x1][x2] = it }
    return dfs(0, 0, grid[0].lastIndex)
  }

```
```rust

  pub fn cherry_pickup(grid: Vec<Vec<i32>>) -> i32 {
    let (h, w, mut ans) = (grid.len(), grid[0].len(), 0);
    let mut dp = vec![vec![vec![-1; w]; w]; h];
    dp[0][0][w - 1] = grid[0][0] + grid[0][w - 1];
    for y in 1..h {
      for x1 in 0..w { for x2 in 0..w {
          let prev = if y > 0 { dp[y - 1][x1][x2] } else { 0 };
          if prev < 0 { continue }
          for d1 in -1..=1 { for d2 in -1..=1 {
              let x1 = x1.wrapping_add(d1 as usize);
              let x2 = x2.wrapping_add(d2 as usize);
              if x1 < x2 && x2 < w {
                let f = prev + grid[y][x1] + grid[y][x2];
                dp[y][x1][x2] = dp[y][x1][x2].max(f);
                ans = ans.max(f);
              }
          }}
      }}
    }
    ans
  }

```

