---
layout: leetcode-entry
title: "576. Out of Boundary Paths"
permalink: "/leetcode/problem/2024-01-26-576-out-of-boundary-paths/"
leetcode_ui: true
entry_slug: "2024-01-26-576-out-of-boundary-paths"
---

[576. Out of Boundary Paths](https://leetcode.com/problems/out-of-boundary-paths/description/) medium
[blog post](https://leetcode.com/problems/out-of-boundary-paths/solutions/4627952/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26012024-576-out-of-boundary-paths?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/SBux3Ye0DDI)
![image.png](/assets/leetcode_daily_images/d170fe61.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/484

#### Problem TLDR

Number of paths from cell in grid to out of boundary.

#### Intuition

Let's do a Brute-Force Depth-First Search from the current cell to neighbors. If we are out of boundary, we have a `1` path, and `0` if moves are out. Then add memoization with a HashMap.

#### Approach

* using `long` helps to shorten the code

#### Complexity

- Time complexity:
$$O(nmv)$$

- Space complexity:
$$O(nmv)$$

#### Code

```kotlin

  fun findPaths(m: Int, n: Int, maxMove: Int, startRow: Int, startColumn: Int): Int {
    val dp = mutableMapOf<Pair<Pair<Int, Int>, Int>, Long>()
    fun dfs(y: Int, x: Int, move: Int): Long = dp.getOrPut(y to x to move) {
      if (y < 0 || x < 0 || y == m || x == n) 1L
      else if (move <= 0) 0L else
      dfs(y - 1, x, move - 1) +
      dfs(y + 1, x, move - 1) +
      dfs(y, x - 1, move - 1) +
      dfs(y, x + 1, move - 1) } % 1_000_000_007L
    return dfs(startRow, startColumn, maxMove).toInt()
  }

```
```rust

  pub fn find_paths(m: i32, n: i32, max_move: i32, start_row: i32, start_column: i32) -> i32 {
      let mut dp = HashMap::new();
      fn dfs( y: i32,  x: i32,  mov: i32,  m: i32,  n: i32,  dp: &mut HashMap<(i32, i32, i32), i64> ) -> i64 {
        if y < 0 || x < 0 || y == m || x == n { 1 } else if mov<= 0 { 0 } else {
            if let Some(&cache) = dp.get(&(y, x, mov)) { cache } else {
              let result = (dfs(y - 1, x, mov - 1, m, n, dp) +
                            dfs(y + 1, x, mov - 1, m, n, dp) +
                            dfs(y, x - 1, mov - 1, m, n, dp) +
                            dfs(y, x + 1, mov - 1, m, n, dp)) % 1_000_000_007;
              dp.insert((y, x, mov), result); result
            }
        }
    }
    dfs(start_row, start_column, max_move, m, n, &mut dp) as i32
  }

```

