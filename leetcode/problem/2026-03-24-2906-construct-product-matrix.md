---
layout: leetcode-entry
title: "2906. Construct Product Matrix"
permalink: "/leetcode/problem/2026-03-24-2906-construct-product-matrix/"
leetcode_ui: true
entry_slug: "2026-03-24-2906-construct-product-matrix"
---

[2906. Construct Product Matrix](https://open.substack.com/pub/dmitriisamoilenko/p/24032026-2906-construct-product-matrix?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[youtube](https://youtu.be/dHZDDOrRz7A)

![24.03.2026.webp](/assets/leetcode_daily_images/24.03.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1307

#### Problem TLDR

Product of other cells in a matrix #medium #prefix

#### Intuition

```j
    // p = a*b*c*d
    // a' = (p/a)%m
    // are we allowed to do 1/a with modulo?
    //
    // corner case: zero if we do %M locally
    //
    // this is some math theory
    //
    // 12345 is not a prime
    // if there is a group of numbers that are multiplies of 12345
    // then all elements are zero except this group (if no duplicates)
    //
    // 12  minute hint: solve without '/', hint2: suffix-prefix
    //
```
Division is not allowed: total product can be 0, and modulo division should be mod inverse, not allowed for 12345

#### Approach

* flatten the matrix to make code shorter

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin [-Kotlin 90ms]
// 90ms
    fun constructProductMatrix(g: Array<IntArray>) = g.flatMap{it.toList()}.run {
        val p = scan(1L) { r, t -> r * t % 12345 }
        val s = reversed().scan(1L) { r, t -> r * t % 12345 }.reversed()
        indices.map { p[it] * s[it+1] % 12345 }.chunked(g[0].size)
    }
```
```rust [-Rust 20ms]
// 20ms
    pub fn construct_product_matrix(g: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let f = g.concat();
        let o = |a: &mut i64, &x: &i32| { let r = *a; *a = *a * x as i64% 12345; Some(r) };
        f.iter().scan(1, o).zip(f.iter().rev().scan(1, o).collect_vec().into_iter().rev())
        .map(|(p,s)| (p * s % 12345) as i32).chunks(g[0].len())
        .into_iter().map(Iterator::collect).collect()
    }
```

