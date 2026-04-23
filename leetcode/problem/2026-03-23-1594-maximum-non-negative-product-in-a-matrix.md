---
layout: leetcode-entry
title: "1594. Maximum Non Negative Product in a Matrix"
permalink: "/leetcode/problem/2026-03-23-1594-maximum-non-negative-product-in-a-matrix/"
leetcode_ui: true
entry_slug: "2026-03-23-1594-maximum-non-negative-product-in-a-matrix"
---

[1594. Maximum Non Negative Product in a Matrix](https://leetcode.com/problems/maximum-non-negative-product-in-a-matrix/solutions/7682943/kotlin-rust-by-samoylenkodmitry-ni8m/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05102025-1594-maximum-non-negative?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/bwOFxmXilFg)

![05.10.2025.webp](/assets/leetcode_daily_images/05.10.2025.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1306

#### Problem TLDR

Max product path right-down #medium #dp

#### Intuition

```j
    // 14minute wrong answer 138 / 159 testcases Output 38431730 Expected 459630706
    // 15 minute tle 152 / 159 testcases passed
    // so the full search is not allowed for 15x15 grid
    // 20 minute, use hint: dp, high+low works
```
The brute-force gives TLE.
Local optimum max is the wrong intuition.
Preserve two results for each cell: max and min.

#### Approach

* only 1D array is needed to track the last row

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(m)$$

#### Code

```kotlin
// 21ms
    fun maxProductPath(g: Array<IntArray>) =
        List(2) { LongArray(g[0].size) }.let { (h, l) ->
            for (y in g.indices) for (x in g[0].indices) {
                if (x + y == 0) { h[x] = 1L*g[y][x]; l[x] = h[x]; continue }
                val i = if (y > 0) x else x-1; val j = if (x > 0) x-1 else x
                val a = listOf(h[i], h[j], l[i], l[j])
                h[x] = a.maxOf { it * g[y][x] }; l[x] = a.minOf { it * g[y][x] }
            }
            h.last().takeIf { it >= 0 }?.let { it%1000000007 } ?: -1L
        }
```
```rust
// 0ms
    pub fn max_product_path(g: Vec<Vec<i32>>) -> i32 {
        let mut h = vec![0i64; g[0].len()]; let mut l = h.clone();
        for y in 0..g.len() { for x in 0..g[0].len() {
            let (v, i, j) = (g[y][x] as _, x-(y==0) as usize, x-(x>0) as usize);
            if x+y<1 { h[x]=v; l[x]=v } else {
            let mut a = [h[i]*v, l[i]*v, h[j]*v, l[j]*v]; a.sort(); h[x] = a[3]; l[x] = a[0] }
        }}
        match *h.last().unwrap() { n if n < 0 => -1, n => (n % 1_000_000_007) as _ }
    }
```

