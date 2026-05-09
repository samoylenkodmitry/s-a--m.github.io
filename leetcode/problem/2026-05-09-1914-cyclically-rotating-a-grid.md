---
layout: leetcode-entry
title: "1914. Cyclically Rotating a Grid"
permalink: "/leetcode/problem/2026-05-09-1914-cyclically-rotating-a-grid/"
leetcode_ui: true
entry_slug: "2026-05-09-1914-cyclically-rotating-a-grid"
---

[1914. Cyclically Rotating a Grid](https://leetcode.com/problems/cyclically-rotating-a-grid/solutions/8176686/kotlin-rust-by-samoylenkodmitry-fub2/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09052026-1914-cyclically-rotating?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/YHTA2lf_Pj8)

https://dmitrysamoylenko.com/leetcode/

![09.05.2026.webp](/assets/leetcode_daily_images/09.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1354

#### Problem TLDR

Rotate matix layers by k

#### Intuition

```j
    // 4x6=16 = 2*3+2*5
    // m,n up to 50, can rotate k times by 1
```
n^3 time, O(1) memory: go layer by layer, then repeat k%p rotates by 1: repeat p times swaps, compute next position
n^2 time, O(1) memory: go layer by layer, linearize each layer by writing get(i) function, do rotation in-place by 3-reversal trick

3-reversal trick: reverse 0..k, k..end, 0..end; the result is shifted left by k

#### Approach

* solutions with O(n) memory are shorter

#### Complexity

- Time complexity:
$$O(n^3|n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun rotateGrid(g: Array<IntArray>, k: Int) = g.also {
        val h = g.size; val w = g[0].size
        for (l in 0..<min(w, h) / 2) {
            val q =  (l..<w - l).map {l to it} +
                     (l + 1..<h - l).map {it to w - 1 - l} +
                     (w - 2 - l downTo l).map {h - 1 - l to it} +
                     (h - 2 - l downTo l + 1).map {it to l }
            val v = q.map { (r, c) -> g[r][c] }
            q.forEachIndexed {i, (r, c) ->  g[r][c] = v[(i + k) % v.size]}
        }
    }
```
```rust
    pub fn rotate_grid(mut g: Vec<Vec<i32>>, k: i32) -> Vec<Vec<i32>> {
        let (w, h, k) = (g[0].len(), g.len(), k as usize);
        for l in 0..w.min(h)/2 {
            let q: Vec<_> = (l..w-l).map(|i| (l,i))
                .chain((l+1..h-l).map(|i|(i,w-1-l)))
                .chain((l..w-1-l).rev().map(|i|(h-1-l,i)))
                .chain((l+1..h-1-l).rev().map(|i|(i,l))).collect();
            let mut rev = |s:usize, e:usize| { for i in 0..(e-s+1)/2 {
                        let ((a,b),(c,d)) = (q[s+i], q[e-i]);
                        let t = g[a][b]; g[a][b] = g[c][d]; g[c][d] = t
                    }};
            rev(0, k%q.len()-1); rev(k%q.len(), q.len()-1); rev(0, q.len()-1)
        }; g
    }
```

