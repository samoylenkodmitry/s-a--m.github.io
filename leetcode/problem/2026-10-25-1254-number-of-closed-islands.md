---
layout: leetcode-entry
title: "1254. Number of Closed Islands"
permalink: "/leetcode/problem/2026-10-25-1254-number-of-closed-islands/"
leetcode_ui: true
entry_slug: "2026-10-25-1254-number-of-closed-islands"
---

[1254. Number of Closed Islands](https://leetcode.com/problems/number-of-closed-islands/solutions/8588075/kotlin-rust-by-samoylenkodmitry-hkg2/) medium
[substack](https://dmitriisamoilenko.substack.com/p/25102026-1254-number-of-closed-islands?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/YAVvQS1uasI)

https://dmitrysamoylenko.com/leetcode/

![25.10.2026.webp](/assets/leetcode_daily_images/25.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1522

#### Problem TLDR

Count 0-islands surrounded by 1

#### Intuition

Do DFS from zeros. Mark visited. Entire island drops if it near the border.

#### Approach

* can modify visited with 1

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun closedIsland(g: Array<IntArray>) = run {
        fun dfs(y: Int, x: Int): Int =
            if (y !in g.indices||x !in g[0].indices) 0 else if (g[y][x]>0) 1
            else (g[y][x]++*0+dfs(y+1,x)*dfs(y-1,x)*dfs(y,x+1)*dfs(y,x-1))
        g.indices.sumOf{y->g[0].indices.count{x->g[y][x]<1&&dfs(y,x)>0}}
    }
```
```rust
    pub fn closed_island(mut g: Vec<Vec<i32>>) -> i32 {
        fn f(g: &mut Vec<Vec<i32>>, y: usize, x: usize) -> i32 {
            if y >= g.len() || x >= g[0].len() { 0 } else if g[y][x] > 0 { 1 }
            else { g[y][x] = 1; f(g, y + 1, x) & f(g, y-1, x) & f(g, y, x + 1) & f(g, y, x-1) }
        }
        (0..g.len()).map(|y| (0..g[0].len()).filter(|&x| g[y][x] < 1 && f(&mut g, y, x) > 0).count() as i32).sum()
    }
```

