---
layout: leetcode-entry
title: "3643. Flip Square Submatrix Vertically"
permalink: "/leetcode/problem/2026-03-21-3643-flip-square-submatrix-vertically/"
leetcode_ui: true
entry_slug: "2026-03-21-3643-flip-square-submatrix-vertically"
---

[3643. Flip Square Submatrix Vertically](https://open.substack.com/pub/dmitriisamoilenko/p/21032026-3643-flip-square-submatrix?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/21032026-3643-flip-square-submatrix?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21032026-3643-flip-square-submatrix?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/XbIS6u_HRVQ)

![c7ed576b-3e7e-4a96-bdcf-e4e9c381c632 (1).webp](/assets/leetcode_daily_images/35531c83.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1304

#### Problem TLDR

Reverse submatrix k by k #easy

#### Intuition

Carefully do this in-place.

#### Approach

* name variables properly
* Rust allows to swap entire slices

#### Complexity

- Time complexity:
$$O(k*k)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 1ms
    fun reverseSubmatrix(g: Array<IntArray>, r: Int, c: Int, k: Int)=g.also{
        for (y in 0..<k/2) for (x in c..<c+k)
            g[y+r][x] = g[r+k-1-y][x].also { g[r+k-1-y][x] = g[y+r][x] }
    }
```
```rust
// 0ms
    pub fn reverse_submatrix(mut g: Vec<Vec<i32>>, r: i32, c: i32, k: i32) -> Vec<Vec<i32>> {
        let (r, c, k) = (r as usize, c as usize, k as usize);
        let (top, btm) = g[r..r+k].split_at_mut(k/2);
        for (t,b) in top.iter_mut().zip(btm.iter_mut().rev()) {
            t[c..c+k].swap_with_slice(&mut b[c..c+k])
        } g
    }
```

