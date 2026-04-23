---
layout: leetcode-entry
title: "1895. Largest Magic Square"
permalink: "/leetcode/problem/2026-01-18-1895-largest-magic-square/"
leetcode_ui: true
entry_slug: "2026-01-18-1895-largest-magic-square"
---

[1895. Largest Magic Square](https://leetcode.com/problems/largest-magic-square/description/) medium
[blog post](https://leetcode.com/problems/largest-magic-square/solutions/7504480/kotlin-rust-by-samoylenkodmitry-6t65/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18012026-1895-largest-magic-square?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/40ioP_-hf0E)

![a0b54509-fbad-4d5b-bbab-028d42549ee3 (1).webp](/assets/leetcode_daily_images/656c6d19.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1241

#### Problem TLDR

Max magic square #medium

#### Intuition

Brute-force is accepted.

#### Approach

* can be optimized with prefix sums: sum(a..b) = p[b]-p[a]

#### Complexity

- Time complexity:
$$O(n^4)$$, or O(n^2) with prefix sums

- Space complexity:
$$O(1)$$, or O(n^2) to store prefix sums

#### Code

```kotlin
// 54ms
    fun largestMagicSquare(g: Array<IntArray>) = (min(g[0].size,g.size) downTo 1)
        .first { s -> (0..g.size-s).any { y -> (0..g[0].size-s).any { x -> val o = 0..<s
            val d =  o.sumOf { g[y+it][x+it] }
                d == o.sumOf { g[y+it][x-it+s-1] } && o.all { i ->
                d == o.sumOf { g[y+i][x+it] } &&
                d == o.sumOf { g[y+it][x+i] }}}}}
```
```rust
// 7ms
    pub fn largest_magic_square(g: Vec<Vec<i32>>) -> i32 {
        (2..=g.len().min(g[0].len())).rev().find(|&s|
            iproduct!(0..=g[0].len()-s, 0..=g.len()-s).any(|(x, y)| {
                let d =  (0..s).map(|i| g[y+i][x+i]).sum::<i32>();
                    d == (0..s).map(|i| g[y+i][x+s-1-i]).sum::<i32>() && (0..s).all(|j|
                    d == (0..s).map(|i| g[y+j][x+i]).sum::<i32>() &&
                    d == (0..s).map(|i| g[y+i][x+j]).sum::<i32>())})).unwrap_or(1) as _
    }
```

