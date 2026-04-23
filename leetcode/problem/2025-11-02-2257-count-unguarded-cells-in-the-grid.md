---
layout: leetcode-entry
title: "2257. Count Unguarded Cells in the Grid"
permalink: "/leetcode/problem/2025-11-02-2257-count-unguarded-cells-in-the-grid/"
leetcode_ui: true
entry_slug: "2025-11-02-2257-count-unguarded-cells-in-the-grid"
---

[2257. Count Unguarded Cells in the Grid](https://leetcode.com/problems/count-unguarded-cells-in-the-grid/description) medium
[blog post](https://leetcode.com/problems/count-unguarded-cells-in-the-grid/solutions/7320840/kotlin-rust-by-samoylenkodmitry-66p0/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02112025-2257-count-unguarded-cells?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2KY_8wP-Zu4)

![21d59441-3b73-4ce0-b535-36863983f07d (1).webp](/assets/leetcode_daily_images/f470d593.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1161

#### Problem TLDR

Count unseen cells by rays from guards #medium #grid

#### Intuition

* 4 rays iterations: left, top, right, bottom
* or, rays from guards until another guard or wall
* or, rays from guards, but mask by direction; don't have to place guards

#### Approach

* count in-place or in another iteration

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 168ms
    fun countUnguarded(m: Int, n: Int, g: Array<IntArray>, w: Array<IntArray>): Int {
        val c = Array(m) { IntArray(n) }; for ((y, x) in g+w) c[y][x] = 2; var x=0; var y=0
        for ((i, j) in g) for (f in setOf({--x;1},{++x;1},{--y;1},{++y;1})) { y=i; x=j; f()
            while (y in 0..<m && x in 0..<n && c[y][x] < 2) c[y][x] = f() }
        return c.sumOf { it.count { it < 1 }}
    }

```
```rust
// 31ms
    pub fn count_unguarded(m: i32, n: i32, g: Vec<Vec<i32>>, w: Vec<Vec<i32>>) -> i32 {
        let mut c = vec![vec![0;n as usize];m as usize]; for g in chain(&g,&w) { c[g[0] as usize][g[1] as usize]=2}
        for g in &g { for (i,j) in &[(-1,0),(0,1),(1,0),(0,-1)] { let (mut y, mut x)=(g[0]+i, g[1]+j);
            while y.min(x)>=0 && y<m && x<n && c[y as usize][x as usize] < 2 { c[y as usize][x as usize]=1; y+=i; x+=j}
        }} c.iter().flatten().filter(|&&v|v<1).count() as _
    }

```

