---
layout: leetcode-entry
title: "3212. Count Submatrices With Equal Frequency of X and Y"
permalink: "/leetcode/problem/2026-03-19-3212-count-submatrices-with-equal-frequency-of-x-and-y/"
leetcode_ui: true
entry_slug: "2026-03-19-3212-count-submatrices-with-equal-frequency-of-x-and-y"
---

[3212. Count Submatrices With Equal Frequency of X and Y](https://open.substack.com/pub/dmitriisamoilenko/p/19032026-3212-count-submatrices-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/19032026-3212-count-submatrices-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19032026-3212-count-submatrices-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/M7-_BA5VpMQ)

![26191ddf-56db-4114-bec4-248112584d32 (1).webp](/assets/leetcode_daily_images/9799c5d4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1302

#### Problem TLDR

Count prefix freq X == freq Y #medium #matrix

#### Intuition

Keep two prefixes: for the X and for the Y.
Freq = freq_left + freq_top

#### Approach

* we can just look at balance x++ y--
* we can store just the last row
* to keep track 'at least one' rule, use a single bit per column

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(m)$$

#### Code

```kotlin
// 18ms
    fun numberOfSubmatrices(g: Array<CharArray>) =
        IntArray(g[0].size).let { X ->
            g.sumOf { r ->
                var x = 0; var s = 0; var j = 0
                r.count { c ->
                    if (c > '.') { s = 1; x += 4*(c - 'X')-2 }
                    X[j] = X[j] + x or s; X[j++] == 1
                }
            }
        }
```
```rust
// 18ms
    pub fn number_of_submatrices(g: Vec<Vec<char>>) -> i32 {
        let mut v = vec![0; g[0].len()];
        g.iter().map(|r| {
            let (mut b, mut s) = (0, 0);
            r.iter().zip(&mut v).map(|(&c, x)| {
                if c > '.' { s = 1; b += (c as i32 - 88)*4-2 }
                *x = *x + b|s; (*x == 1) as i32
            }).sum::<i32>()
        }).sum()
    }
```

