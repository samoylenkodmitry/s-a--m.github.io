---
layout: leetcode-entry
title: "3882. Minimum XOR Path in a Grid"
permalink: "/leetcode/problem/2026-10-28-3882-minimum-xor-path-in-a-grid/"
leetcode_ui: true
entry_slug: "2026-10-28-3882-minimum-xor-path-in-a-grid"
---

[3882. Minimum XOR Path in a Grid](https://leetcode.com/problems/minimum-xor-path-in-a-grid/solutions/8592387/kotlin-rust-by-samoylenkodmitry-wrtm/) medium
[substack](https://dmitriisamoilenko.substack.com/p/28102026-3882-minimum-xor-path-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/bRTN6rxtDtQ)

https://dmitrysamoylenko.com/leetcode/

![28.10.2026.webp](/assets/leetcode_daily_images/28.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1525

#### Problem TLDR

Smallest xor-path

#### Intuition

This is a simple Dp problem. Cache all possible xors for all possible paths. There could not be more than 1023 paths totally.

#### Approach

* first cell for the bottom-up or that last cell for the top-down is the first xor value

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun minCost(g: Array<IntArray>) = run {
        val dp = HashMap<Int, Set<Int>>()
        fun f(y: Int, x: Int): Set<Int> = dp.getOrPut(y*1000+x) {
            ((if (y+1<g.size) f(y+1,x) else setOf())+if (x+1<g[0].size) f(y,x+1) else setOf())
            .ifEmpty { setOf(0) }.map { it xor g[y][x] }.toSet()
        }
        f(0, 0).min()
    }
```
```rust
    pub fn min_cost(g: Vec<Vec<i32>>) -> i32 {
        let mut d = [[false; 1024]; 1001]; d[0][0] = true;
        for r in &g { for (x, &v) in (0..).zip(r) {
            d[x + 1] = from_fn(|k| d[x][k ^ v as usize] || d[x + 1][k ^ v as usize]);
            d[0][0] = false
        }}
        d[g[0].len()].iter().position(|&b| b).unwrap() as _
    }
```

